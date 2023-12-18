using I = ILGPU.Index1D;
using Data = ILGPU.ArrayView<int>;

namespace ILGPUTest;

public static class CopyArrayDataTest
{
    public static void Run()
    {
        using var context = Context.CreateDefault();
        using var gpu = context.CreateCPUAccelerator(0);

        //using var context = Context.Create(b => b.Cuda());
        //using var gpu     = context.CreateCudaAccelerator(0);

        //using var context = Context.Create(b => b.OpenCL());
        //using var gpu     = context.CreateCLAccelerator(0);

        int[] data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        var gpu_data = gpu.Allocate1D(data);
        var gpu_out  = gpu.Allocate1D<int>(10_000);

        var kernel = gpu.LoadAutoGroupedStreamKernel(static (I i, Data data, Data output) => output[i] = data[i % data.Length]);

        kernel((int)gpu_out.Length, gpu_data.View, gpu_out.View);

        gpu.Synchronize();

        var output = gpu_out.GetAsArray1D();
    }
}
