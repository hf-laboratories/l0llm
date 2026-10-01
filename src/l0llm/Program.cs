using System.Diagnostics;
using System.Globalization;
using System.Text;
using HFLabs.ML.LLM;
using HFLabs.ML.LLM.LevelZero;
using HFLabs.ML.LLM.LevelZero.Cli;
using LevelZero;
using LevelZero.Kernels;

// l0llm: chat / run / bench / check / info for local HuggingFace models on an Intel GPU through Level Zero.
return await Cli.RunAsync(args).ConfigureAwait(false);

internal static class Cli
{
    private const string Usage = """
        l0llm - local HuggingFace LLMs on the Intel GPU (Level Zero)

        Usage:
          l0llm info
          l0llm check [--model DIR]                 smoke test: load, generate a few tokens, print speed
          l0llm chat  [--model DIR] [options]       interactive multi-turn chat (type /exit to quit, /reset to clear)
          l0llm run   [--model DIR] --prompt TEXT   one-shot generation
          l0llm bench [--model DIR] [--precision P] load time, prefill and decode speed
          l0llm serve [--model DIR] [--port 11434]  OpenAI-compatible REST API server (SSE streaming, CORS)

        Options:
          --model DIR        model directory (config.json, tokenizer.json, model.safetensors).
                             Default: $L0LLM_MODEL, else the first model found under $L0LLM_MODELS or .\models
          --port PORT        server port for l0llm serve (default: 11434)
          --host HOST        server host for l0llm serve (default: localhost)
          --precision P      int8 (default, DP4A), fp16 or fp32
          --max-new N        maximum new tokens (default 512; 16 for check)
          --max-seq N        KV cache length (default 4096)
          --temp T           sampling temperature; 0 = greedy (default 0.7)
          --top-p P / --top-k K / --repeat R   sampling controls
          --system TEXT      system prompt for chat/run
          --raw              run: send the prompt as-is (no chat template)
          --think/--no-think Qwen3, Qwen3.5: force thinking mode on or off (default: the model's own)
          --beams N          run: beam search with N beams instead of sampling (128 new tokens by default)
          --n-best K         with --beams: print the K best completions (default 1)
          --length-penalty A with --beams: score = logprob / length^A (default 1.0)
        """;

    public static async Task<int> RunAsync(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            Console.WriteLine(Usage);
            return args.Length == 0 ? 1 : 0;
        }

        string command = args[0].ToLowerInvariant();
        Dictionary<string, string> opts;
        try
        {
            opts = ParseOptions(args.Skip(1).ToArray());
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 2;
        }

        try
        {
            return command switch
            {
                "info" => Info(),
                "check" => await CheckAsync(opts).ConfigureAwait(false),
                "chat" => await ChatAsync(opts).ConfigureAwait(false),
                "run" => await RunOnceAsync(opts).ConfigureAwait(false),
                "bench" => Bench(opts),
                "serve" => await ServeAsync(opts).ConfigureAwait(false),
                _ => Fail($"Unknown command '{command}'.\n\n{Usage}"),
            };
        }
        catch (Exception ex) when (ex is InvalidDataException or NotSupportedException or FileNotFoundException or DirectoryNotFoundException or InvalidOperationException)
        {
            Console.Error.WriteLine($"error: {ex.Message}");
            return 1;
        }
    }

    private static int Fail(string message)
    {
        Console.Error.WriteLine(message);
        return 2;
    }

    private static Dictionary<string, string> ParseOptions(string[] args)
    {
        var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            if (!a.StartsWith("--", StringComparison.Ordinal))
            {
                throw new ArgumentException($"Unexpected argument '{a}'.");
            }

            string key = a[2..];
            if (key is "raw" or "think" or "no-think")
            {
                map[key] = "true";
                continue;
            }

            if (i + 1 >= args.Length)
            {
                throw new ArgumentException($"Option '{a}' needs a value.");
            }

            map[key] = args[++i];
        }

        return map;
    }

    private static int Info()
    {
        bool available;
        try
        {
            available = LevelZeroRuntime.IsAvailable();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Level Zero: not available ({ex.Message})");
            return 1;
        }

        Console.WriteLine($"Level Zero: {(available ? "available" : "NOT available (install the Intel graphics driver)")}");
        if (!available)
        {
            return 1;
        }

        foreach (DeviceInfo d in LevelZeroRuntime.EnumerateDevices())
        {
            Console.WriteLine($"  device {d.DriverIndex}.{d.DeviceIndex}: {d.Name}");
        }

        foreach (string dir in FindModels())
        {
            Console.WriteLine($"  model: {dir}");
        }

        return 0;
    }

    private static IEnumerable<string> FindModels()
    {
        var roots = new List<string>();
        string? env = Environment.GetEnvironmentVariable("L0LLM_MODELS");
        if (!string.IsNullOrWhiteSpace(env))
        {
            roots.Add(env);
        }

        roots.Add(Path.Combine(AppContext.BaseDirectory, "models"));
        roots.Add(Path.Combine(Directory.GetCurrentDirectory(), "models"));
        foreach (string root in roots.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (!Directory.Exists(root))
            {
                continue;
            }

            foreach (string dir in Directory.EnumerateDirectories(root).OrderBy(d => d, StringComparer.OrdinalIgnoreCase))
            {
                if ((File.Exists(Path.Combine(dir, "model.safetensors")) ||
                     File.Exists(Path.Combine(dir, "model.safetensors.index.json")) ||
                     Directory.EnumerateFiles(dir, "*.safetensors").Any()) &&
                    File.Exists(Path.Combine(dir, "config.json")))
                {
                    yield return dir;
                }
            }
        }
    }

    private static string ResolveModel(Dictionary<string, string> opts)
    {
        string? dir = opts.GetValueOrDefault("model") ?? Environment.GetEnvironmentVariable("L0LLM_MODEL");
        dir ??= FindModels().FirstOrDefault();
        if (string.IsNullOrEmpty(dir))
        {
            throw new FileNotFoundException("No model found. Pass --model DIR or put model folders under .\\models.");
        }

        if (!File.Exists(Path.Combine(dir, "model.safetensors")) &&
            !File.Exists(Path.Combine(dir, "model.safetensors.index.json")) &&
            !Directory.EnumerateFiles(dir, "*.safetensors").Any())
        {
            throw new FileNotFoundException($"'{dir}' does not contain model.safetensors or model.safetensors.index.json.");
        }

        return Path.GetFullPath(dir);
    }

    private static LlmWeightPrecision ParsePrecision(Dictionary<string, string> opts) =>
        opts.GetValueOrDefault("precision")?.ToLowerInvariant() switch
        {
            null or "int8" or "auto" => LlmWeightPrecision.Auto,
            "fp16" or "f16" or "half" => LlmWeightPrecision.Float16,
            "fp32" or "f32" or "float" => LlmWeightPrecision.Float32,
            string p => throw new ArgumentException($"Unknown precision '{p}' (use int8, fp16 or fp32)."),
        };

    private static int IntOpt(Dictionary<string, string> opts, string key, int fallback) =>
        opts.TryGetValue(key, out string? v) ? int.Parse(v, CultureInfo.InvariantCulture) : fallback;

    private static float FloatOpt(Dictionary<string, string> opts, string key, float fallback) =>
        opts.TryGetValue(key, out string? v) ? float.Parse(v, CultureInfo.InvariantCulture) : fallback;

    private static LevelZeroLlmEngine CreateEngine(string modelDir, Dictionary<string, string> opts, out string modelId)
    {
        string parent = Path.GetDirectoryName(Path.TrimEndingDirectorySeparator(modelDir))!;
        modelId = Path.GetFileName(Path.TrimEndingDirectorySeparator(modelDir));
        var options = new LlmOptions { ModelDirectory = parent, Backend = LlmOptions.LevelZeroNativeBackend };
        return new LevelZeroLlmEngine(options, maxSeqLen: IntOpt(opts, "max-seq", 4096), weightPrecision: ParsePrecision(opts));
    }

    private static LlmGenerationOptions Generation(Dictionary<string, string> opts, int defaultMax) => new()
    {
        Temperature = FloatOpt(opts, "temp", 0.7f),
        TopP = FloatOpt(opts, "top-p", 0.9f),
        TopK = IntOpt(opts, "top-k", 40),
        RepetitionPenalty = FloatOpt(opts, "repeat", 1.1f),
        MaxNewTokens = IntOpt(opts, "max-new", defaultMax),
    };

    private static async Task<int> CheckAsync(Dictionary<string, string> opts)
    {
        string dir = ResolveModel(opts);
        Console.WriteLine($"model: {dir}");
        using LevelZeroLlmEngine engine = CreateEngine(dir, opts, out string id);
        var watch = Stopwatch.StartNew();
        var sb = new StringBuilder();
        int pieces = 0;
        LlmGenerationOptions gen = new() { Temperature = 0f, MaxNewTokens = IntOpt(opts, "max-new", 16) };
        double firstMs = 0;
        await foreach (string piece in engine.GenerateAsync("The capital of France is", id, gen).ConfigureAwait(false))
        {
            if (pieces++ == 0)
            {
                firstMs = watch.Elapsed.TotalMilliseconds;
            }

            _ = sb.Append(piece);
        }

        double total = watch.Elapsed.TotalSeconds;
        Console.WriteLine($"output: {sb}");
        Console.WriteLine($"first token after {firstMs / 1000:F1}s (includes model load), {pieces} pieces in {total:F1}s");
        if (pieces == 0)
        {
            Console.Error.WriteLine("FAIL: no output produced");
            return 1;
        }

        Console.WriteLine("OK");
        return 0;
    }

    private static async Task<int> RunOnceAsync(Dictionary<string, string> opts)
    {
        string prompt = opts.GetValueOrDefault("prompt") ?? throw new ArgumentException("--prompt is required.");
        string dir = ResolveModel(opts);
        using LevelZeroLlmEngine engine = CreateEngine(dir, opts, out string id);
        string text = opts.ContainsKey("raw") ? prompt : ChatTemplates.ForModelDirectory(dir).Format(BuildTurns(opts, [], prompt), ThinkingOptions(opts));
        if (IntOpt(opts, "beams", 1) > 1)
        {
            int beams = IntOpt(opts, "beams", 1);
            var search = new BeamSearchOptions
            {
                BeamWidth = beams,
                NumReturn = Math.Min(IntOpt(opts, "n-best", 1), beams),
                MaxNewTokens = IntOpt(opts, "max-new", 128),
                LengthPenalty = FloatOpt(opts, "length-penalty", 1.0f),
            };
            var watch = Stopwatch.StartNew();
            IReadOnlyList<BeamCompletion> results = await engine.GenerateBeamAsync(text, id, search).ConfigureAwait(false);
            for (int i = 0; i < results.Count; i++)
            {
                BeamHypothesis h = results[i].Hypothesis;
                if (results.Count > 1)
                {
                    Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"--- #{i + 1} score {h.Score:F3} logprob {h.LogProb:F2} ({h.Tokens.Length} tokens) ---"));
                }

                Console.WriteLine(results[i].Text);
            }

            Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture, $"({beams} beams, {watch.Elapsed.TotalSeconds:F1}s)"));
            return 0;
        }

        await foreach (string piece in engine.GenerateAsync(text, id, Generation(opts, 512)).ConfigureAwait(false))
        {
            Console.Write(piece);
        }

        Console.WriteLine();
        return 0;
    }

    /// <summary>--think / --no-think switch Qwen3 and Qwen3.5 thinking mode; without either, each model keeps its default.</summary>
    private static ChatTemplateOptions ThinkingOptions(Dictionary<string, string> opts) =>
        new() { EnableThinking = opts.ContainsKey("no-think") ? false : opts.ContainsKey("think") ? true : null };

    private static List<ChatTurn> BuildTurns(Dictionary<string, string> opts, List<ChatTurn> history, string user)
    {
        var turns = new List<ChatTurn>();
        if (opts.TryGetValue("system", out string? system))
        {
            turns.Add(new ChatTurn("system", system));
        }

        turns.AddRange(history);
        turns.Add(new ChatTurn("user", user));
        return turns;
    }

    private static async Task<int> ChatAsync(Dictionary<string, string> opts)
    {
        string dir = ResolveModel(opts);
        Console.WriteLine($"Loading {Path.GetFileName(dir)} ...");
        using LevelZeroLlmEngine engine = CreateEngine(dir, opts, out string id);
        IChatTemplate template = ChatTemplates.ForModelDirectory(dir);
        var history = new List<ChatTurn>();
        Console.WriteLine("Type a message. /reset clears the conversation, /exit quits.");
        while (true)
        {
            Console.Write("\nyou> ");
            string? line = Console.ReadLine();
            if (line is null || line.Trim() is "/exit" or "/quit")
            {
                return 0;
            }

            if (line.Trim() == "/reset")
            {
                history.Clear();
                Console.WriteLine("(conversation cleared)");
                continue;
            }

            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            string prompt = template.Format(BuildTurns(opts, history, line), ThinkingOptions(opts));
            var reply = new StringBuilder();
            Console.Write("bot> ");
            var watch = Stopwatch.StartNew();
            int pieces = 0;
            await foreach (string piece in engine.GenerateAsync(prompt, id, Generation(opts, 512)).ConfigureAwait(false))
            {
                pieces++;
                Console.Write(piece);
                _ = reply.Append(piece);
            }

            Console.WriteLine();
            Console.WriteLine($"({pieces} pieces, {watch.Elapsed.TotalSeconds:F1}s)");
            history.Add(new ChatTurn("user", line));
            history.Add(new ChatTurn("assistant", reply.ToString()));
        }
    }

    private static async Task<int> ServeAsync(Dictionary<string, string> opts)
    {
        string dir = ResolveModel(opts);
        Console.WriteLine($"Initializing Level Zero LLM engine with model: {Path.GetFileName(dir)} ...");
        using LevelZeroLlmEngine engine = CreateEngine(dir, opts, out string id);
        return await L0LlmServer.StartAsync(dir, opts, engine, id).ConfigureAwait(false);
    }

    private static int Bench(Dictionary<string, string> opts)
    {
        string dir = ResolveModel(opts);
        using ComputeDevice device = LevelZeroRuntime.GetDefaultDevice();
        using LlmKernelSuite suite = LlmKernelSuite.Create(device);
        var watch = Stopwatch.StartNew();
        using LlmModel model = LlmModel.Load(device, suite, dir, maxSeqLen: 256, precision: ParsePrecision(opts));
        double load = watch.Elapsed.TotalSeconds;
        int[] prompt = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198];
        _ = model.Prefill(prompt);
        _ = model.Decode(785);
        watch.Restart();
        _ = model.Prefill(prompt);
        double prefillMs = watch.Elapsed.TotalMilliseconds;
        const int steps = 48;
        int token = 785;
        watch.Restart();
        for (int i = 0; i < steps; i++)
        {
            float[] logits = model.Decode(token);
            token = Array.IndexOf(logits, logits.Max());
        }

        double decodeMs = watch.Elapsed.TotalMilliseconds;
        Console.WriteLine(string.Create(
            CultureInfo.InvariantCulture,
            $"{Path.GetFileName(dir)} [{(model.UsesInt8Weights ? "int8" : model.UsesHalfWeights ? "fp16" : "fp32")}] load {load:F1}s, prefill {prefillMs / prompt.Length:F1} ms/token, decode {decodeMs / steps:F1} ms/token = {steps * 1000.0 / decodeMs:F1} tokens/s"));
        return 0;
    }
}
