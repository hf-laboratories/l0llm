using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace HFLabs.ML.LLM
{
    public interface ILlmEngine
    {
        IAsyncEnumerable<string> GenerateAsync(
            string prompt,
            string modelId,
            LlmGenerationOptions options,
            CancellationToken ct = default);
    }

    public sealed class LlmGenerationOptions
    {
        public float Temperature { get; init; } = 0.7f;
        public float TopP { get; init; } = 0.9f;
        public int TopK { get; init; } = 40;
        public float RepetitionPenalty { get; init; } = 1.1f;
        public int MaxNewTokens { get; init; } = 256;
        public string[] StopSequences { get; init; } = Array.Empty<string>();
    }

    public sealed class LlmOptions
    {
        public string ModelDirectory { get; set; } = string.Empty;
        public string TokenizerJsonPath { get; set; } = string.Empty;
        public const string LevelZeroNativeBackend = "LevelZeroNative";
        public string Backend { get; set; } = "LevelZeroNative";
        public int DeviceId { get; set; } = 0;
    }

    public sealed class LlmSampler
    {
        private readonly Random _random = new();

        public int Sample(
            float[] logits,
            float temperature = 0.7f,
            float topP = 0.9f,
            int topK = 40,
            float repetitionPenalty = 1.1f,
            IReadOnlyList<int>? contextTokens = null)
        {
            if (logits == null || logits.Length == 0)
            {
                throw new ArgumentException("Logits array cannot be null or empty.", nameof(logits));
            }

            int vocabSize = logits.Length;
            float[] logitsCopy = new float[vocabSize];
            Array.Copy(logits, logitsCopy, vocabSize);

            // 1. Repetition Penalty
            if (repetitionPenalty != 1.0f && contextTokens != null && contextTokens.Count > 0)
            {
                var uniqueContext = new HashSet<int>(contextTokens);
                foreach (int tokenId in uniqueContext)
                {
                    if (tokenId >= 0 && tokenId < vocabSize)
                    {
                        float logit = logitsCopy[tokenId];
                        if (logit >= 0.0f)
                        {
                            logitsCopy[tokenId] = logit / repetitionPenalty;
                        }
                        else
                        {
                            logitsCopy[tokenId] = logit * repetitionPenalty;
                        }
                    }
                }
            }

            // 2. Greedy search (T <= 0)
            if (temperature <= 0.0f)
            {
                int maxIdx = 0;
                float maxVal = logitsCopy[0];
                for (int i = 1; i < vocabSize; i++)
                {
                    if (logitsCopy[i] > maxVal)
                    {
                        maxVal = logitsCopy[i];
                        maxIdx = i;
                    }
                }
                return maxIdx;
            }

            // 3. Apply Temperature scaling
            for (int i = 0; i < vocabSize; i++)
            {
                logitsCopy[i] /= temperature;
            }

            // 4. Softmax
            float maxLogit = logitsCopy.Max();
            double sum = 0.0;
            double[] probs = new double[vocabSize];
            for (int i = 0; i < vocabSize; i++)
            {
                probs[i] = Math.Exp(logitsCopy[i] - maxLogit);
                sum += probs[i];
            }
            for (int i = 0; i < vocabSize; i++)
            {
                probs[i] /= sum;
            }

            // 5. Top-K / Top-P
            var indexedProbs = new List<(int Index, double Prob)>(vocabSize);
            for (int i = 0; i < vocabSize; i++)
            {
                indexedProbs.Add((i, probs[i]));
            }

            indexedProbs.Sort((a, b) => b.Prob.CompareTo(a.Prob));

            if (topK > 0 && topK < indexedProbs.Count)
            {
                indexedProbs.RemoveRange(topK, indexedProbs.Count - topK);
            }

            if (topP > 0.0f && topP < 1.0f)
            {
                double cumulativeProb = 0.0;
                int keepCount = 0;
                for (int i = 0; i < indexedProbs.Count; i++)
                {
                    cumulativeProb += indexedProbs[i].Prob;
                    keepCount++;
                    if (cumulativeProb >= topP)
                    {
                        break;
                    }
                }
                if (keepCount < indexedProbs.Count)
                {
                    indexedProbs.RemoveRange(keepCount, indexedProbs.Count - keepCount);
                }
            }

            double filterSum = indexedProbs.Sum(p => p.Prob);
            if (filterSum <= 0.0)
            {
                return indexedProbs[0].Index;
            }

            double randomValue = _random.NextDouble() * filterSum;
            double currentSum = 0.0;
            for (int i = 0; i < indexedProbs.Count; i++)
            {
                currentSum += indexedProbs[i].Prob;
                if (randomValue <= currentSum)
                {
                    return indexedProbs[i].Index;
                }
            }

            return indexedProbs[^1].Index;
        }
    }
}
