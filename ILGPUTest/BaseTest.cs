namespace ILGPUTest;

public static class BaseTest
{
    public static void Run()
    {
        using var context = Context.CreateDefault();
        using var accelerator = context.CreateCPUAccelerator(0);

        //using var context = Context.Create(b => b.Cuda());
        //using var accelerator = context.CreateCudaAccelerator(0);

        //using var context     = Context.Create(b => b.OpenCL());
        //using var accelerator = context.CreateCLAccelerator(0);

        var device_data   = accelerator.Allocate1D(new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        var device_output = accelerator.Allocate1D<int>(10_000);

        var kernel = accelerator.LoadAutoGroupedStreamKernel((Index1D i, ArrayView<int> data, ArrayView<int> output) => output[i] = data[i % data.Length]);

        kernel((int)device_output.Length, device_data.View, device_output.View);

        accelerator.Synchronize();

        var output = device_output.GetAsArray1D();
    }
}
