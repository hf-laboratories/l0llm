using Xunit;

// Each model fixture pins one or more GB of shared (USM) memory on the iGPU; running fixtures for
// several checkpoints at once exhausts it (zeMemAllocShared fails), so test classes run one at a time.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
