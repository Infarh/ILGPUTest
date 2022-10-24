using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ILGPUTest;

public static class ArrayMultiplyTest
{
    public static void Run()
    {
        using var context     = Context.CreateDefault();
        using var accelerator = context.CreateCudaAccelerator(0);

        const int count = 100_000_000;

        var a = Enumerable.Range(1, count).ToArray();
        var b = a.Select(v => v * 10).ToArray();

        Console.WriteLine("Data created");

        var device_a = accelerator.Allocate1D(a);
        var device_b = accelerator.Allocate1D(b);
        var device_c = accelerator.Allocate1D<int>(8_000);

        Console.WriteLine("GPU initialized");

        static void Kernel(Index1D i, ArrayView<int> A, ArrayView<int> B, ArrayView<int> C) => C[i] = A[i] * B[i];

        var kernel = accelerator.LoadAutoGroupedStreamKernel((Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>)Kernel);

        var timer = Stopwatch.StartNew();
        kernel((int)device_c.Length, device_a.View, device_b.View, device_c.View);

        Console.WriteLine("GPU started");

        accelerator.Synchronize();

        timer.Stop();

        Console.WriteLine("GPU completed");

        var elapsed_gpu = timer.Elapsed;

        var expected_c = new int[count];

        Console.WriteLine("CPU started");

        timer.Restart();
        for (var i = 0; i < count; i++)
            expected_c[i] = a[i] * b[i];
        timer.Stop();

        Console.WriteLine("CPU completed");

        var elapsed_cpu = timer.Elapsed;

        var c = device_c.GetAsArray1D();

        Console.WriteLine("Elapsed CPU: {0}", elapsed_cpu);
        Console.WriteLine("Elapsed GPU: {0}", elapsed_gpu);
    }
}
