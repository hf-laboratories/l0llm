using System.Text.Json;
using LevelZero;
using LevelZero.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace HFLabs.ML.LLM.LevelZero.Tests;

/// <summary>
/// M3 gate: one Qwen2.5-0.5B-Instruct decoder layer on the GPU must reproduce HuggingFace
/// transformers (float32, eager attention) on a 15-token prompt. Reference values come from
/// <c>golden/gen_layer_golden.py</c>. Each token is fed the reference layer input, so errors do not
/// compound across layers or tokens.
/// Skips silently without a GPU or the checkpoint; set <c>IPU_L0_REQUIRE_HARDWARE=1</c> and
/// <c>HF_REQUIRE_MODEL=1</c> to make either a failure.
/// </summary>
public sealed class Qwen25LayerGoldenTests
{
    private const double AbsTolerance = 1e-3;
    private const double RelTolerance = 1e-3;

    private static readonly string ModelDir =
        Environment.GetEnvironmentVariable("HF_QWEN25_05B_INSTRUCT_DIR") ?? @"D:\models\Qwen2.5-0.5B-Instruct";

    private static readonly Lazy<JsonElement> Golden = new(() =>
    {
        string path = Path.Combine(AppContext.BaseDirectory, "golden", "qwen2.5-0.5b-instruct.layers.golden.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.Clone();
    });

    private readonly ITestOutputHelper _output;

    public Qwen25LayerGoldenTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(0)]
    [InlineData(11)]
    [InlineData(22)]
    public void Layer_MatchesTransformersOnEveryTokenAndStage(int layer)
    {
        if (!Available())
        {
            return;
        }

        JsonElement golden = Golden.Value;
        JsonElement layerGolden = golden.GetProperty("layers").GetProperty(layer.ToString(System.Globalization.CultureInfo.InvariantCulture));
        HfModelConfig config = HfModelConfig.Load(ModelDir);
        float[][] input = Rows(layerGolden.GetProperty("input"));
        int tokens = input.Length;
        Assert.Equal(golden.GetProperty("token_ids").GetArrayLength(), tokens);

        var captured = new Dictionary<string, List<float[]>>();
        using var checkpoint = SafetensorsCheckpoint.Open(ModelDir);
        ComputeDevice device = LevelZeroRuntime.GetDefaultDevice();
        using (device)
        using (LlmKernelSuite suite = LlmKernelSuite.Create(device))
        using (LlmLayerWeights weights = LlmLayerWeights.Load(device, checkpoint, config, layer))
        using (var layerRunner = new LlmDecoderLayer(device, suite, config, weights))
        using (var cache = new LlmKvCache(device, maxSeqLen: 64, config.KeyValueDim))
        using (SharedBuffer<float> hidden = device.AllocShared<float>(config.HiddenSize))
        {
            layerRunner.Probe = (stage, buffer) =>
            {
                if (!captured.TryGetValue(stage, out List<float[]>? list))
                {
                    captured[stage] = list = [];
                }

                list.Add(buffer.ToArray());
            };

            for (int t = 0; t < tokens; t++)
            {
                hidden.Write(input[t]);
                layerRunner.Forward(hidden, t, cache);
            }
        }

        var stages = new (string Name, string GoldenKey)[]
        {
            ("input_norm", "input_norm"), ("q", "q"), ("k", "k"), ("v", "v"), ("attn_out", "attn_out"),
            ("post_norm", "post_norm"), ("mlp_out", "mlp_out"), ("output", "output"),
        };
        var failures = new List<string>();
        foreach ((string name, string key) in stages)
        {
            float[][] expected = Rows(layerGolden.GetProperty(key));
            List<float[]> actual = captured[name];
            Assert.Equal(tokens, actual.Count);

            double maxAbs = 0;
            double maxExcess = 0;
            for (int t = 0; t < tokens; t++)
            {
                Assert.Equal(expected[t].Length, actual[t].Length);
                for (int i = 0; i < expected[t].Length; i++)
                {
                    double diff = Math.Abs((double)expected[t][i] - actual[t][i]);
                    maxAbs = Math.Max(maxAbs, diff);
                    maxExcess = Math.Max(maxExcess, diff - (AbsTolerance + (RelTolerance * Math.Abs(expected[t][i]))));
                }
            }

            _output.WriteLine($"layer {layer} {name}: max abs error {maxAbs:E3}");
            if (maxExcess > 0)
            {
                failures.Add($"{name}: max abs error {maxAbs:E3} exceeds {AbsTolerance} + {RelTolerance} * |expected|");
            }
        }

        Assert.True(failures.Count == 0, $"Layer {layer}: " + string.Join("; ", failures));
    }

    private static bool Available()
    {
        bool hardware;
        try
        {
            hardware = LevelZeroRuntime.IsAvailable();
        }
        catch (Exception)
        {
            hardware = false;
        }

        bool model = File.Exists(Path.Combine(ModelDir, "model.safetensors"));
        if (!hardware && Environment.GetEnvironmentVariable("IPU_L0_REQUIRE_HARDWARE") == "1")
        {
            throw new InvalidOperationException("IPU_L0_REQUIRE_HARDWARE=1 but no Level Zero device was detected.");
        }

        if (!model && Environment.GetEnvironmentVariable("HF_REQUIRE_MODEL") == "1")
        {
            throw new InvalidOperationException($"HF_REQUIRE_MODEL=1 but no checkpoint at '{ModelDir}'.");
        }

        return hardware && model;
    }

    private static float[][] Rows(JsonElement tensor)
    {
        int rows = tensor.GetProperty("shape")[0].GetInt32();
        int cols = tensor.GetProperty("shape")[1].GetInt32();
        byte[] bytes = Convert.FromBase64String(tensor.GetProperty("data").GetString()!);
        Assert.Equal(rows * cols * 4, bytes.Length);
        var result = new float[rows][];
        for (int r = 0; r < rows; r++)
        {
            result[r] = new float[cols];
            Buffer.BlockCopy(bytes, r * cols * 4, result[r], 0, cols * 4);
        }

        return result;
    }
}
