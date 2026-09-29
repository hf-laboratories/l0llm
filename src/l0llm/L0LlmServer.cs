using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using HFLabs.ML.LLM;
using HFLabs.ML.LLM.LevelZero;

namespace HFLabs.ML.LLM.LevelZero.Cli;

internal static class L0LlmServer
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public static async Task<int> StartAsync(string modelDir, Dictionary<string, string> opts, LevelZeroLlmEngine engine, string modelId)
    {
        int port = opts.TryGetValue("port", out string? portStr) && int.TryParse(portStr, CultureInfo.InvariantCulture, out int p) ? p : 11434;
        string host = opts.GetValueOrDefault("host") ?? "localhost";
        string prefix = $"http://{host}:{port}/";

        using var listener = new HttpListener();
        try
        {
            listener.Prefixes.Add(prefix);
            listener.Start();
        }
        catch (HttpListenerException ex)
        {
            Console.Error.WriteLine($"Failed to bind to {prefix}: {ex.Message}");
            if (host == "localhost")
            {
                prefix = $"http://127.0.0.1:{port}/";
                listener.Prefixes.Clear();
                listener.Prefixes.Add(prefix);
                listener.Start();
            }
            else
            {
                throw;
            }
        }

        Console.WriteLine($"==================================================");
        Console.WriteLine($" l0llm OpenAI-compatible Server listening on {prefix}");
        Console.WriteLine($" Model loaded: {modelId}");
        Console.WriteLine($" Endpoints:");
        Console.WriteLine($"   GET  /v1/models");
        Console.WriteLine($"   POST /v1/chat/completions");
        Console.WriteLine($"   POST /v1/completions");
        Console.WriteLine($"   GET  /health");
        Console.WriteLine($" Connect any UI (Open-WebUI, Chatbox, Continue.dev) to this base URL.");
        Console.WriteLine($" Press Ctrl+C to stop.");
        Console.WriteLine($"==================================================");

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            cts.Cancel();
            try { listener.Stop(); } catch { }
        };

        var sem = new SemaphoreSlim(1, 1); // Serialized GPU inference queue

        while (!cts.Token.IsCancellationRequested)
        {
            HttpListenerContext context;
            try
            {
                context = await listener.GetContextAsync().WaitAsync(cts.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception)
            {
                if (!listener.IsListening) break;
                continue;
            }

            _ = Task.Run(() => HandleRequestAsync(context, engine, modelId, sem, cts.Token));
        }

        Console.WriteLine("\nServer shutting down...");
        return 0;
    }

    private static async Task HandleRequestAsync(
        HttpListenerContext context,
        LevelZeroLlmEngine engine,
        string modelId,
        SemaphoreSlim sem,
        CancellationToken ct)
    {
        HttpListenerRequest req = context.Request;
        HttpListenerResponse res = context.Response;

        // Enable CORS for all web UIs
        res.AddHeader("Access-Control-Allow-Origin", "*");
        res.AddHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.AddHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With");

        if (req.HttpMethod.Equals("OPTIONS", StringComparison.OrdinalIgnoreCase))
        {
            res.StatusCode = (int)HttpStatusCode.OK;
            res.Close();
            return;
        }

        string path = req.Url?.AbsolutePath.TrimEnd('/') ?? "";

        try
        {
            if (req.HttpMethod.Equals("GET", StringComparison.OrdinalIgnoreCase))
            {
                if (path is "" or "/health")
                {
                    await WriteJsonAsync(res, new { status = "ok", engine = "l0llm", backend = "LevelZero", model = modelId }).ConfigureAwait(false);
                    return;
                }

                if (path == "/v1/models")
                {
                    var response = new
                    {
                        @object = "list",
                        data = new[]
                        {
                            new
                            {
                                id = modelId,
                                @object = "model",
                                created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                                owned_by = "l0llm"
                            }
                        }
                    };
                    await WriteJsonAsync(res, response).ConfigureAwait(false);
                    return;
                }
            }
            else if (req.HttpMethod.Equals("POST", StringComparison.OrdinalIgnoreCase))
            {
                if (path == "/v1/chat/completions")
                {
                    using var reader = new StreamReader(req.InputStream, req.ContentEncoding);
                    string body = await reader.ReadToEndAsync(ct).ConfigureAwait(false);
                    ChatCompletionRequest? chatReq = JsonSerializer.Deserialize<ChatCompletionRequest>(body, JsonOpts);

                    if (chatReq is null || chatReq.Messages is null || chatReq.Messages.Count == 0)
                    {
                        res.StatusCode = (int)HttpStatusCode.BadRequest;
                        await WriteJsonAsync(res, new { error = "Invalid or empty messages array." }).ConfigureAwait(false);
                        return;
                    }

                    // Format turns into ChatML prompt
                    var turns = new List<ChatTurn>();
                    foreach (var m in chatReq.Messages)
                    {
                        turns.Add(new ChatTurn(m.Role ?? "user", m.Content ?? ""));
                    }
                    string prompt = QwenChatTemplate.Format(turns);

                    var genOptions = new LlmGenerationOptions
                    {
                        Temperature = chatReq.Temperature ?? 0.7f,
                        TopP = chatReq.TopP ?? 0.9f,
                        TopK = chatReq.TopK ?? 40,
                        RepetitionPenalty = chatReq.RepetitionPenalty ?? 1.1f,
                        MaxNewTokens = chatReq.MaxTokens ?? chatReq.MaxCompletionTokens ?? 512,
                    };

                    await sem.WaitAsync(ct).ConfigureAwait(false);
                    try
                    {
                        if (chatReq.Stream)
                        {
                            await HandleStreamChatAsync(res, engine, modelId, prompt, genOptions, ct).ConfigureAwait(false);
                        }
                        else
                        {
                            await HandleNonStreamChatAsync(res, engine, modelId, prompt, genOptions, ct).ConfigureAwait(false);
                        }
                    }
                    finally
                    {
                        sem.Release();
                    }
                    return;
                }

                if (path == "/v1/completions")
                {
                    using var reader = new StreamReader(req.InputStream, req.ContentEncoding);
                    string body = await reader.ReadToEndAsync(ct).ConfigureAwait(false);
                    CompletionRequest? compReq = JsonSerializer.Deserialize<CompletionRequest>(body, JsonOpts);

                    if (compReq is null || string.IsNullOrEmpty(compReq.Prompt))
                    {
                        res.StatusCode = (int)HttpStatusCode.BadRequest;
                        await WriteJsonAsync(res, new { error = "Invalid or empty prompt." }).ConfigureAwait(false);
                        return;
                    }

                    var genOptions = new LlmGenerationOptions
                    {
                        Temperature = compReq.Temperature ?? 0.7f,
                        TopP = compReq.TopP ?? 0.9f,
                        MaxNewTokens = compReq.MaxTokens ?? 512,
                    };

                    await sem.WaitAsync(ct).ConfigureAwait(false);
                    try
                    {
                        if (compReq.Stream)
                        {
                            await HandleStreamChatAsync(res, engine, modelId, compReq.Prompt, genOptions, ct).ConfigureAwait(false);
                        }
                        else
                        {
                            await HandleNonStreamChatAsync(res, engine, modelId, compReq.Prompt, genOptions, ct).ConfigureAwait(false);
                        }
                    }
                    finally
                    {
                        sem.Release();
                    }
                    return;
                }
            }

            res.StatusCode = (int)HttpStatusCode.NotFound;
            await WriteJsonAsync(res, new { error = $"Path '{path}' not found." }).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            try
            {
                res.StatusCode = (int)HttpStatusCode.InternalServerError;
                await WriteJsonAsync(res, new { error = ex.Message }).ConfigureAwait(false);
            }
            catch { }
        }
        finally
        {
            try { res.Close(); } catch { }
        }
    }

    private static async Task HandleNonStreamChatAsync(
        HttpListenerResponse res,
        LevelZeroLlmEngine engine,
        string modelId,
        string prompt,
        LlmGenerationOptions gen,
        CancellationToken ct)
    {
        var sb = new StringBuilder();
        int completionTokens = 0;
        await foreach (string piece in engine.GenerateAsync(prompt, modelId, gen, ct).ConfigureAwait(false))
        {
            completionTokens++;
            sb.Append(piece);
        }

        string id = $"chatcmpl-{Guid.NewGuid():N}";
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        var response = new
        {
            id,
            @object = "chat.completion",
            created,
            model = modelId,
            choices = new[]
            {
                new
                {
                    index = 0,
                    message = new { role = "assistant", content = sb.ToString() },
                    finish_reason = "stop"
                }
            },
            usage = new
            {
                prompt_tokens = prompt.Length / 4, // estimate
                completion_tokens = completionTokens,
                total_tokens = (prompt.Length / 4) + completionTokens
            }
        };

        await WriteJsonAsync(res, response).ConfigureAwait(false);
    }

    private static async Task HandleStreamChatAsync(
        HttpListenerResponse res,
        LevelZeroLlmEngine engine,
        string modelId,
        string prompt,
        LlmGenerationOptions gen,
        CancellationToken ct)
    {
        res.ContentType = "text/event-stream; charset=utf-8";
        res.StatusCode = (int)HttpStatusCode.OK;
        res.SendChunked = true;

        string id = $"chatcmpl-{Guid.NewGuid():N}";
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        using var writer = new StreamWriter(res.OutputStream, new UTF8Encoding(false), bufferSize: 1024, leaveOpen: true);

        await foreach (string piece in engine.GenerateAsync(prompt, modelId, gen, ct).ConfigureAwait(false))
        {
            var chunk = new
            {
                id,
                @object = "chat.completion.chunk",
                created,
                model = modelId,
                choices = new[]
                {
                    new
                    {
                        index = 0,
                        delta = new { content = piece },
                        finish_reason = (string?)null
                    }
                }
            };

            string json = JsonSerializer.Serialize(chunk, JsonOpts);
            await writer.WriteAsync($"data: {json}\n\n").ConfigureAwait(false);
            await writer.FlushAsync(ct).ConfigureAwait(false);
        }

        // Send final finish chunk
        var finalChunk = new
        {
            id,
            @object = "chat.completion.chunk",
            created,
            model = modelId,
            choices = new[]
            {
                new
                {
                    index = 0,
                    delta = new { },
                    finish_reason = "stop"
                }
            }
        };
        string finalJson = JsonSerializer.Serialize(finalChunk, JsonOpts);
        await writer.WriteAsync($"data: {finalJson}\n\n").ConfigureAwait(false);
        await writer.WriteAsync("data: [DONE]\n\n").ConfigureAwait(false);
        await writer.FlushAsync(ct).ConfigureAwait(false);
    }

    private static async Task WriteJsonAsync(HttpListenerResponse res, object payload)
    {
        res.ContentType = "application/json; charset=utf-8";
        byte[] bytes = JsonSerializer.SerializeToUtf8Bytes(payload, JsonOpts);
        res.ContentLength64 = bytes.Length;
        await res.OutputStream.WriteAsync(bytes).ConfigureAwait(false);
    }

    private sealed class ChatCompletionRequest
    {
        public string? Model { get; set; }
        public List<ChatMessage>? Messages { get; set; }
        public bool Stream { get; set; }
        public float? Temperature { get; set; }
        [JsonPropertyName("top_p")]
        public float? TopP { get; set; }
        [JsonPropertyName("top_k")]
        public int? TopK { get; set; }
        [JsonPropertyName("repetition_penalty")]
        public float? RepetitionPenalty { get; set; }
        [JsonPropertyName("max_tokens")]
        public int? MaxTokens { get; set; }
        [JsonPropertyName("max_completion_tokens")]
        public int? MaxCompletionTokens { get; set; }
    }

    private sealed class ChatMessage
    {
        public string? Role { get; set; }
        public string? Content { get; set; }
    }

    private sealed class CompletionRequest
    {
        public string? Model { get; set; }
        public string? Prompt { get; set; }
        public bool Stream { get; set; }
        public float? Temperature { get; set; }
        [JsonPropertyName("top_p")]
        public float? TopP { get; set; }
        [JsonPropertyName("max_tokens")]
        public int? MaxTokens { get; set; }
    }
}
