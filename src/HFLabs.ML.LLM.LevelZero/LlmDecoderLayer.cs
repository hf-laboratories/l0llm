using LevelZero;
using LevelZero.Kernels;

namespace HFLabs.ML.LLM.LevelZero;

/// <summary>
/// Runs one decoder layer on the GPU: input RMSNorm, a token mixer, residual + RMSNorm, gated MLP, residual.
/// The mixer is either full attention (Q/K/V projections with optional bias, per-head QK-norm and output gate,
/// RoPE, KV-cache store, attention over the cache, output projection) or, for Qwen3.5, a Gated DeltaNet
/// (projections, causal conv, gated delta rule, gated norm, output projection).
/// </summary>
/// <remarks>
/// <see cref="Forward"/> handles one token at an explicit position and <see cref="ForwardBatch"/> a chunk of up to
/// <see cref="BatchCapacity"/> consecutive tokens. The fused attention kernel keeps all scores in local memory,
/// which limits the attended length to <see cref="MaxAttendableTokens"/>. A Gated DeltaNet layer keeps a
/// <see cref="LlmLinearState"/> instead of a KV cache; it must see the tokens in order and is reset when a call starts at
/// position 0.
/// </remarks>
public sealed class LlmDecoderLayer : IDisposable
{
    /// <summary>Longest sequence the attention kernel's local-memory score buffer supports.</summary>
    public const int MaxAttendableTokens = 15_000;

    private readonly ComputeDevice _device;
    private readonly LlmKernelSuite _suite;
    private readonly HfModelConfig _config;
    private readonly LlmLayerWeights _weights;
    private readonly SharedBuffer<float> _normedInput;
    private readonly SharedBuffer<float> _q;
    private readonly SharedBuffer<float> _k;
    private readonly SharedBuffer<float> _v;
    private readonly SharedBuffer<float> _attention;
    private readonly SharedBuffer<float> _attentionProjection;
    private readonly SharedBuffer<float> _normedPost;
    private readonly SharedBuffer<float> _gate;
    private readonly SharedBuffer<float> _up;
    private readonly SharedBuffer<float> _down;
    private readonly SharedBuffer<int> _position;
    private readonly SharedBuffer<float>? _attentionGate;
    private readonly SharedBuffer<float>? _inverseFrequencies;
    private readonly float _ropeAttentionScale = 1f;
    private BatchBuffers? _batch;
    private LinearBuffers? _linearBuffers;

    /// <summary>Creates a layer runner. The runner does not own <paramref name="suite"/> or <paramref name="weights"/>.</summary>
    /// <exception cref="ArgumentException">The config needs QK-norm, an output gate or bias weights the layer does not carry.</exception>
    public LlmDecoderLayer(ComputeDevice device, LlmKernelSuite suite, HfModelConfig config, LlmLayerWeights weights)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(suite);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        config.EnsureSupported();
        IsLinear = weights.Linear is not null;
        if (!IsLinear)
        {
            if (weights.QProj is null || weights.KProj is null || weights.VProj is null || weights.OProj is null)
            {
                throw new ArgumentException("The layer weights carry no attention projections.", nameof(weights));
            }

            if (config.QkNorm && (weights.QNorm is null || weights.KNorm is null))
            {
                throw new ArgumentException("The config has QK-norm but the layer weights carry none.", nameof(weights));
            }

            if (config.AttentionBias && (weights.QBias is null || weights.KBias is null || weights.VBias is null))
            {
                throw new ArgumentException("The config has attention bias but the layer weights carry none.", nameof(weights));
            }

            if (config.AttentionOutputGate && weights.AttentionGateProj is null)
            {
                throw new ArgumentException("The config has an attention output gate but the layer weights carry none.", nameof(weights));
            }
        }

        _device = device;
        _suite = suite;
        _config = config;
        _weights = weights;
        _normedInput = device.AllocShared<float>(config.HiddenSize);
        _q = device.AllocShared<float>(config.QueryDim);
        _k = device.AllocShared<float>(config.KeyValueDim);
        _v = device.AllocShared<float>(config.KeyValueDim);
        _attention = device.AllocShared<float>(config.QueryDim);
        _attentionProjection = device.AllocShared<float>(config.HiddenSize);
        _normedPost = device.AllocShared<float>(config.HiddenSize);
        _gate = device.AllocShared<float>(config.IntermediateSize);
        _up = device.AllocShared<float>(config.IntermediateSize);
        _down = device.AllocShared<float>(config.HiddenSize);
        _position = device.AllocShared<int>(1);
        if (config.AttentionOutputGate && !IsLinear)
        {
            _attentionGate = device.AllocShared<float>(config.QueryDim);
        }

        if (!IsLinear && config.NeedsRopeTable)
        {
            (float[] freqs, float attentionScale) = RopeFrequencies.Compute(config);
            _inverseFrequencies = device.AllocShared<float>(freqs.Length);
            _inverseFrequencies.Write(freqs);
            _ropeAttentionScale = attentionScale;
        }
    }

    /// <summary>True when this layer is a Gated DeltaNet layer (it needs a <see cref="LlmLinearState"/>, not a KV cache).</summary>
    public bool IsLinear { get; }

    /// <summary>Most tokens <see cref="ForwardBatch"/> accepts in one call.</summary>
    public int BatchCapacity { get; } = LlmGemvKernel.MaxRows;

    /// <summary>
    /// Optional callback invoked with a stage name and the buffer that holds that stage's result, right
    /// after the stage finished. Stages: <c>input_norm</c>, <c>q</c>, <c>k</c>, <c>v</c> (after bias, before
    /// RoPE; full attention only), <c>attn_out</c> (after the mixer's output projection), <c>post_norm</c>,
    /// <c>mlp_out</c>, <c>output</c>. The buffer is reused, so copy what you need inside the callback.
    /// </summary>
    public Action<string, SharedBuffer<float>>? Probe { get; set; }

    /// <summary>
    /// Runs the layer for one token. <paramref name="hidden"/> holds the layer input on entry and the
    /// layer output on return.
    /// </summary>
    /// <param name="hidden">Residual stream for this token [hiddenSize], updated in place.</param>
    /// <param name="position">Absolute position of this token; also its slot in the KV cache.</param>
    /// <param name="cache">This layer's KV cache (full attention). Slots 0..position-1 must already be filled.</param>
    /// <param name="linearState">This layer's Gated DeltaNet state (linear attention); reset when <paramref name="position"/> is 0.</param>
    public void Forward(SharedBuffer<float> hidden, int position, LlmKvCache? cache, LlmLinearState? linearState = null)
    {
        ArgumentNullException.ThrowIfNull(hidden);
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        if (hidden.Count < _config.HiddenSize)
        {
            throw new ArgumentException($"hidden holds {hidden.Count} elements, need {_config.HiddenSize}.", nameof(hidden));
        }

        if (IsLinear)
        {
            ArgumentNullException.ThrowIfNull(linearState);
        }
        else
        {
            ArgumentNullException.ThrowIfNull(cache);
            if (position >= cache.MaxSeqLen)
            {
                throw new ArgumentOutOfRangeException(nameof(position), position, $"Cache holds {cache.MaxSeqLen} positions.");
            }

            if (position + 1 > MaxAttendableTokens)
            {
                throw new NotSupportedException(
                    $"Attention over {position + 1} tokens exceeds the kernel's local-memory limit of {MaxAttendableTokens}.");
            }

            if (cache.KvDim != _config.KeyValueDim)
            {
                throw new ArgumentException($"Cache row width {cache.KvDim} does not match {_config.KeyValueDim}.", nameof(cache));
            }
        }

        int h = _config.HiddenSize;
        int inter = _config.IntermediateSize;
        LlmKernelSuite s = _suite;

        // 1. Input RMSNorm (x is left untouched: no residual add).
        s.RMSNorm.Execute(_device, hidden, null, _weights.InputNorm, _normedInput, _config.RmsNormEps, h, 1, addResidual: false);
        Emit("input_norm", _normedInput);

        // 2. Token mixer.
        if (IsLinear)
        {
            MixLinear(linearState!, position == 0, 1, _normedInput, _attentionProjection);
        }
        else
        {
            MixFull(position, cache!);
        }

        Emit("attn_out", _attentionProjection);

        // 3. hidden += mixer output, and post-attention RMSNorm of the sum.
        s.RMSNorm.Execute(
            _device, hidden, _attentionProjection, _weights.PostNorm, _normedPost, _config.RmsNormEps, h, 1, addResidual: true);
        Emit("post_norm", _normedPost);

        // 4. Gated MLP: down(silu(gate(x)) * up(x)).
        LlmMatrix.ProjectShared(_device, s, 1, _normedPost, _weights.GateProj, _gate, _weights.UpProj, _up);
        s.SwiGLU.Execute(_device, _gate, _up, inter);
        _weights.DownProj.Project(_device, s, _gate, _down);
        Emit("mlp_out", _down);

        // 5. hidden += MLP output.
        s.Add.Execute(_device, hidden, _down, h);
        Emit("output", hidden);
    }

    /// <summary>
    /// Runs the layer for a chunk of <paramref name="tokens"/> consecutive prompt tokens (positions
    /// <paramref name="position0"/> .. <paramref name="position0"/> + tokens - 1) in one pass, so every weight matrix is
    /// read once per <see cref="LlmGemvKernel.MaxRows"/> tokens instead of once per token. Attention is causal within
    /// the chunk and covers everything already in the cache.
    /// </summary>
    /// <param name="hidden">Residual stream for the chunk [tokens, hiddenSize], updated in place.</param>
    /// <param name="tokens">Number of tokens in the chunk.</param>
    /// <param name="position0">Absolute position of the first token; also its slot in the KV cache.</param>
    /// <param name="cache">This layer's KV cache (full attention). Slots 0..position0-1 must already be filled.</param>
    /// <param name="linearState">This layer's Gated DeltaNet state (linear attention); reset when <paramref name="position0"/> is 0.</param>
    public void ForwardBatch(SharedBuffer<float> hidden, int tokens, int position0, LlmKvCache? cache, LlmLinearState? linearState = null)
    {
        ArgumentNullException.ThrowIfNull(hidden);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(tokens);
        ArgumentOutOfRangeException.ThrowIfNegative(position0);
        if (tokens > BatchCapacity)
        {
            throw new ArgumentOutOfRangeException(nameof(tokens), tokens, $"At most {BatchCapacity} tokens per chunk.");
        }

        if ((long)hidden.Count < (long)tokens * _config.HiddenSize)
        {
            throw new ArgumentException($"hidden holds {hidden.Count} elements, need {(long)tokens * _config.HiddenSize}.", nameof(hidden));
        }

        if (IsLinear)
        {
            ArgumentNullException.ThrowIfNull(linearState);
        }
        else
        {
            ArgumentNullException.ThrowIfNull(cache);
            if (position0 + tokens > cache.MaxSeqLen)
            {
                throw new ArgumentOutOfRangeException(nameof(position0), position0, $"Cache holds {cache.MaxSeqLen} positions.");
            }

            if (position0 + tokens > MaxAttendableTokens)
            {
                throw new NotSupportedException(
                    $"Attention over {position0 + tokens} tokens exceeds the kernel's local-memory limit of {MaxAttendableTokens}.");
            }

            if (cache.KvDim != _config.KeyValueDim)
            {
                throw new ArgumentException($"Cache row width {cache.KvDim} does not match {_config.KeyValueDim}.", nameof(cache));
            }
        }

        BatchBuffers b = _batch ??= new BatchBuffers(_device, _config, BatchCapacity);
        int h = _config.HiddenSize;
        int inter = _config.IntermediateSize;
        LlmKernelSuite s = _suite;
        s.RMSNorm.Execute(_device, hidden, null, _weights.InputNorm, b.NormedInput, _config.RmsNormEps, h, tokens, addResidual: false);
        if (IsLinear)
        {
            MixLinear(linearState!, position0 == 0, tokens, b.NormedInput, b.AttentionProjection);
        }
        else
        {
            MixFullBatch(b, tokens, position0, cache!);
        }

        s.RMSNorm.Execute(
            _device, hidden, b.AttentionProjection, _weights.PostNorm, b.NormedPost, _config.RmsNormEps, h, tokens, addResidual: true);
        LlmMatrix.ProjectShared(_device, s, tokens, b.NormedPost, _weights.GateProj, b.Gate, _weights.UpProj, b.Up);
        s.SwiGLU.Execute(_device, b.Gate, b.Up, tokens * inter);
        _weights.DownProj.ProjectRows(_device, s, tokens, b.Gate, b.Down);
        s.Add.Execute(_device, hidden, b.Down, tokens * h);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _inverseFrequencies?.Dispose();
        _attentionGate?.Dispose();
        _batch?.Dispose();
        _linearBuffers?.Dispose();
        _normedInput.Dispose();
        _q.Dispose();
        _k.Dispose();
        _v.Dispose();
        _attention.Dispose();
        _attentionProjection.Dispose();
        _normedPost.Dispose();
        _gate.Dispose();
        _up.Dispose();
        _down.Dispose();
        _position.Dispose();
    }

    // Full attention for one token: Q/K/V (+ bias, QK-norm, output gate), RoPE, cache store, attention, output projection.
    private void MixFull(int position, LlmKvCache cache)
    {
        LlmKernelSuite s = _suite;
        int qDim = _config.QueryDim;
        int kvDim = _config.KeyValueDim;
        LlmMatrix.ProjectShared(
            _device, s, 1, _normedInput,
            _weights.QProj!, _q, _weights.KProj!, _k, _weights.VProj!, _v,
            _attentionGate is not null ? _weights.AttentionGateProj! : null, _attentionGate);

        if (_config.AttentionBias)
        {
            s.BiasAdd.Execute(_device, _q, _weights.QBias!, qDim, 1);
            s.BiasAdd.Execute(_device, _k, _weights.KBias!, kvDim, 1);
            s.BiasAdd.Execute(_device, _v, _weights.VBias!, kvDim, 1);
        }

        if (_config.QkNorm)
        {
            // Qwen3: per-head RMSNorm on Q and K (shared [headDim] weight), before RoPE.
            s.RMSNorm.Execute(_device, _q, null, _weights.QNorm!, _q, _config.RmsNormEps, _config.HeadDim, _config.NumAttentionHeads, addResidual: false);
            s.RMSNorm.Execute(_device, _k, null, _weights.KNorm!, _k, _config.RmsNormEps, _config.HeadDim, _config.NumKeyValueHeads, addResidual: false);
        }

        Emit("q", _q);
        Emit("k", _k);
        Emit("v", _v);

        // Rotary embeddings on Q and K, then store this token's K/V in the cache.
        _position.Write([position]);
        ApplyRope(_q, _k, _position, 1);
        s.KvCacheStore.Execute(_device, _k, _v, cache.Keys, cache.Values, position, kvDim);

        // Attention over slots 0..position, the optional sigmoid output gate, then the output projection.
        s.Attention.Execute(
            _device, _q, cache.Keys, cache.Values, _attention, _config.HeadDim, _config.NumAttentionHeads,
            _config.NumKeyValueHeads, position + 1, cache.MaxSeqLen);
        if (_attentionGate is not null)
        {
            s.SigmoidGate.Execute(_device, _attention, _attentionGate, qDim);
        }

        _weights.OProj!.Project(_device, s, _attention, _attentionProjection);
    }

    // Full attention for a chunk of tokens.
    private void MixFullBatch(BatchBuffers b, int tokens, int position0, LlmKvCache cache)
    {
        LlmKernelSuite s = _suite;
        int qDim = _config.QueryDim;
        int kvDim = _config.KeyValueDim;
        var positions = new int[tokens];
        for (int i = 0; i < tokens; i++)
        {
            positions[i] = position0 + i;
        }

        b.Positions.Write(positions);
        LlmMatrix.ProjectShared(
            _device, s, tokens, b.NormedInput,
            _weights.QProj!, b.Q, _weights.KProj!, b.K, _weights.VProj!, b.V,
            _config.AttentionOutputGate ? _weights.AttentionGateProj! : null, b.AttentionGate);

        if (_config.AttentionBias)
        {
            s.BiasAdd.Execute(_device, b.Q, _weights.QBias!, qDim, tokens);
            s.BiasAdd.Execute(_device, b.K, _weights.KBias!, kvDim, tokens);
            s.BiasAdd.Execute(_device, b.V, _weights.VBias!, kvDim, tokens);
        }

        if (_config.QkNorm)
        {
            s.RMSNorm.Execute(_device, b.Q, null, _weights.QNorm!, b.Q, _config.RmsNormEps, _config.HeadDim, tokens * _config.NumAttentionHeads, addResidual: false);
            s.RMSNorm.Execute(_device, b.K, null, _weights.KNorm!, b.K, _config.RmsNormEps, _config.HeadDim, tokens * _config.NumKeyValueHeads, addResidual: false);
        }

        ApplyRope(b.Q, b.K, b.Positions, tokens);
        s.KvCacheStore.ExecuteRows(_device, b.K, b.V, cache.Keys, cache.Values, position0, kvDim, tokens);
        s.Attention.ExecutePrefill(
            _device, b.Q, cache.Keys, cache.Values, b.Attention, _config.HeadDim, _config.NumAttentionHeads,
            _config.NumKeyValueHeads, position0, tokens, cache.MaxSeqLen);
        if (_config.AttentionOutputGate)
        {
            s.SigmoidGate.Execute(_device, b.Attention, b.AttentionGate!, tokens * qDim);
        }

        _weights.OProj!.ProjectRows(_device, s, tokens, b.Attention, b.AttentionProjection);
    }

    // Gated DeltaNet for one token or a chunk of tokens, in order. Writes [tokens, hidden] to output.
    private void MixLinear(LlmLinearState state, bool reset, int tokens, SharedBuffer<float> input, SharedBuffer<float> output)
    {
        LlmKernelSuite s = _suite;
        LlmLinearAttentionWeights lw = _weights.Linear!;
        LinearBuffers lb = _linearBuffers ??= new LinearBuffers(_device, _config, BatchCapacity);
        if (reset)
        {
            state.Reset();
        }

        LlmMatrix.ProjectShared(_device, s, tokens, input, lw.InProjQkv, lb.Qkv, lw.InProjZ, lb.Z, lw.InProjB, lb.B, lw.InProjA, lb.A);
        s.CausalConv.Execute(_device, lb.Qkv, state.Conv, lw.Conv, lb.Conv, _config.LinearConvDim, _config.LinearConvKernel, tokens);
        s.GatedDelta.Execute(
            _device, lb.Conv, lb.Z, lb.B, lb.A, lw.ALog, lw.DtBias, lw.Norm, state.Recurrent, lb.Mixed,
            _config.LinearValueHeads, _config.LinearKeyHeadDim, _config.LinearValueHeadDim, tokens, _config.RmsNormEps);
        Project(lw.OutProj, tokens, lb.Mixed, output);
    }

    private void Project(LlmMatrix matrix, int tokens, SharedBuffer<float> x, SharedBuffer<float> y)
    {
        if (tokens == 1)
        {
            matrix.Project(_device, _suite, x, y);
        }
        else
        {
            matrix.ProjectRows(_device, _suite, tokens, x, y);
        }
    }

    private void ApplyRope(SharedBuffer<float> q, SharedBuffer<float> k, SharedBuffer<int> positions, int tokens)
    {
        if (_inverseFrequencies is null)
        {
            _suite.RoPE.Execute(
                _device, q, k, positions, _config.HeadDim, _config.NumAttentionHeads, _config.NumKeyValueHeads,
                (float)_config.RopeTheta, tokens);
        }
        else
        {
            _suite.RoPE.ExecuteWithFrequencies(
                _device, q, k, positions, _inverseFrequencies, _config.HeadDim, _config.NumAttentionHeads,
                _config.NumKeyValueHeads, _ropeAttentionScale, tokens, _config.RotaryDim);
        }
    }

    private void Emit(string stage, SharedBuffer<float> buffer) => Probe?.Invoke(stage, buffer);

    /// <summary>Scratch buffers for <see cref="ForwardBatch"/>, allocated on first use.</summary>
    private sealed class BatchBuffers : IDisposable
    {
        private readonly List<IDisposable> _owned = [];

        public BatchBuffers(ComputeDevice device, HfModelConfig c, int capacity)
        {
            NormedInput = Own(device.AllocShared<float>(capacity * c.HiddenSize));
            Q = Own(device.AllocShared<float>(capacity * c.QueryDim));
            K = Own(device.AllocShared<float>(capacity * c.KeyValueDim));
            V = Own(device.AllocShared<float>(capacity * c.KeyValueDim));
            Attention = Own(device.AllocShared<float>(capacity * c.QueryDim));
            AttentionProjection = Own(device.AllocShared<float>(capacity * c.HiddenSize));
            NormedPost = Own(device.AllocShared<float>(capacity * c.HiddenSize));
            Gate = Own(device.AllocShared<float>(capacity * c.IntermediateSize));
            Up = Own(device.AllocShared<float>(capacity * c.IntermediateSize));
            Down = Own(device.AllocShared<float>(capacity * c.HiddenSize));
            Positions = Own(device.AllocShared<int>(capacity));
            if (c.AttentionOutputGate)
            {
                AttentionGate = Own(device.AllocShared<float>(capacity * c.QueryDim));
            }
        }

        public SharedBuffer<float> NormedInput { get; }

        public SharedBuffer<float> Q { get; }

        public SharedBuffer<float> K { get; }

        public SharedBuffer<float> V { get; }

        public SharedBuffer<float> Attention { get; }

        public SharedBuffer<float> AttentionProjection { get; }

        public SharedBuffer<float> NormedPost { get; }

        public SharedBuffer<float> Gate { get; }

        public SharedBuffer<float> Up { get; }

        public SharedBuffer<float> Down { get; }

        public SharedBuffer<int> Positions { get; }

        public SharedBuffer<float>? AttentionGate { get; }

        public void Dispose()
        {
            foreach (IDisposable d in _owned)
            {
                d.Dispose();
            }
        }

        private SharedBuffer<T> Own<T>(SharedBuffer<T> buffer)
            where T : unmanaged
        {
            _owned.Add(buffer);
            return buffer;
        }
    }

    /// <summary>Scratch buffers for a Gated DeltaNet layer, sized for a chunk of <see cref="BatchCapacity"/> tokens.</summary>
    private sealed class LinearBuffers : IDisposable
    {
        private readonly List<IDisposable> _owned = [];

        public LinearBuffers(ComputeDevice device, HfModelConfig c, int capacity)
        {
            Qkv = Own(device.AllocShared<float>(capacity * c.LinearConvDim));
            Conv = Own(device.AllocShared<float>(capacity * c.LinearConvDim));
            Z = Own(device.AllocShared<float>(capacity * c.LinearValueDim));
            B = Own(device.AllocShared<float>(capacity * c.LinearValueHeads));
            A = Own(device.AllocShared<float>(capacity * c.LinearValueHeads));
            Mixed = Own(device.AllocShared<float>(capacity * c.LinearValueDim));
        }

        public SharedBuffer<float> Qkv { get; }

        public SharedBuffer<float> Conv { get; }

        public SharedBuffer<float> Z { get; }

        public SharedBuffer<float> B { get; }

        public SharedBuffer<float> A { get; }

        public SharedBuffer<float> Mixed { get; }

        public void Dispose()
        {
            foreach (IDisposable d in _owned)
            {
                d.Dispose();
            }
        }

        private SharedBuffer<float> Own(SharedBuffer<float> buffer)
        {
            _owned.Add(buffer);
            return buffer;
        }
    }
}
