using System.Diagnostics;
using Array2D = ILGPU.Runtime.ArrayView2D<float, ILGPU.Stride2D.DenseX>;
using I2D = ILGPU.Index2D;
using Dx = ILGPU.Stride2D.DenseX;

namespace ILGPUTest;

public static class MatrixMultiplyTest
{
    private const int __N = 500;
    private const int __M = 700;
    private const int __K = 600;

    private static float[,] InitRandom(float[,] M)
    {
        var (n, m) = (M.GetLength(0), M.GetLength(1));
        var rnd = Random.Shared;

        for (var i = 0; i < n; i++)
            for (var j = 0; j < m; j++)
                M[i, j] = (float)rnd.NextDouble();

        return M;
    }

    public static void RunSimple()
    {
        float[,] A =
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 },
            { 10, 11, 12 },
        };

        float[,] B =
        {
            { 13, 14, 15, 16, 17 },
            { 18, 19, 20, 21, 22 },
            { 23, 24, 25, 26, 27 },
        };

        float[,] C =
        {
            { 118, 124, 130, 136, 142 },
            { 280, 295, 310, 325, 340 },
            { 442, 466, 490, 514, 538 },
            { 604, 637, 670, 703, 736 },
        };

        Console.WriteLine("Started");
        var timer = Stopwatch.StartNew();

        A = InitRandom(new float[__N, __M]);
        B = InitRandom(new float[__M, __K]);

        Console.WriteLine("Initialized {0}", timer.Elapsed);
        timer.Restart();

        var m = A.GetLength(0);
        var k = A.GetLength(1);
        var n = B.GetLength(1);

        using var context     = Context.CreateDefault();
        using var accelerator = context.CreateCudaAccelerator(0);

        var kernel = accelerator.LoadAutoGroupedStreamKernel<I2D, Array2D, Array2D, Array2D>(MatrixMultiplyAcceleratedKernel);

        Console.WriteLine("Kernel created {0}", timer.Elapsed);
        timer.Restart();

        static void MatrixMultiplyAcceleratedKernel(I2D IndexIJ, Array2D a, Array2D b, Array2D result)
        {
            var s_ij = 0.0f;

            var i = IndexIJ.X;
            var j = IndexIJ.Y;
            for (var k = 0; k < a.IntExtent.Y; k++)
                s_ij += a[new(i, k)] * b[new(k, j)];

            result[IndexIJ] = s_ij;
        }

        Console.Write("Copy data to GPU ");

        using var a_buffer      = accelerator.Allocate2DDenseX<float>(new(m, k));
        using var b_buffer      = accelerator.Allocate2DDenseX<float>(new(k, n));
        using var result_buffer = accelerator.Allocate2DDenseX<float>(new(m, n));
        a_buffer.CopyFromCPU(A);
        b_buffer.CopyFromCPU(B);

        Console.WriteLine(timer.Elapsed);

        Console.WriteLine("Begin");
        timer.Restart();

        kernel(result_buffer.Extent.ToIntIndex(), a_buffer.View, b_buffer.View, result_buffer.View);
        accelerator.Synchronize();

        Console.WriteLine("Computed {0}", timer.Elapsed);

        Console.Write("Copy data from GPU ");

        var actual_c = result_buffer.GetAsArray2D();
        Console.WriteLine(timer.Elapsed);

        Console.WriteLine("MatrixMultiplyAcceleratedKernel completed");

        Console.WriteLine();
        RunTiled();
    }

    public static void RunTiled()
    {
        float[,] A =
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 },
            { 10, 11, 12 },
        };

        float[,] B =
        {
            { 13, 14, 15, 16, 17 },
            { 18, 19, 20, 21, 22 },
            { 23, 24, 25, 26, 27 },
        };

        float[,] C =
        {
            { 118, 124, 130, 136, 142 },
            { 280, 295, 310, 325, 340 },
            { 442, 466, 490, 514, 538 },
            { 604, 637, 670, 703, 736 },
        };

        Console.WriteLine("Started");

        var timer = Stopwatch.StartNew();
        A = InitRandom(new float[__N, __M]);
        B = InitRandom(new float[__M, __K]);

        Console.WriteLine("Initialized {0}", timer.Elapsed);
        timer.Restart();

        var m = A.GetLength(0);
        var k = A.GetLength(1);
        var n = B.GetLength(1);

        using var context     = Context.CreateDefault();
        using var accelerator = context.CreateCudaAccelerator(0);

        var kernel = accelerator.LoadStreamKernel<Array2D, Array2D, Array2D>(MatrixMultiplyTiledKernel);

        Console.WriteLine("Kernel created {0}", timer.Elapsed);

        const int TILE_SIZE = 32; // accelerator.Device.MemoryBusWidth == 64 -> TILE_SIZE = accelerator.Device.MemoryBusWidth / sizeof(float) = 32
        if (TILE_SIZE > accelerator.Device.MemoryBusWidth / (sizeof(float) / 2))
            throw new InvalidOperationException(
                $"Размер тайла ({TILE_SIZE}*sizeof(float)/2 = {TILE_SIZE * sizeof(float) / 2} байт) не может быть больше ширины пропускания шины данных устройства ({accelerator.Device.MemoryBusWidth} байт)");

        static void MatrixMultiplyTiledKernel(Array2D a, Array2D b, Array2D result)
        {
            var global   = Grid.GlobalIndex.XY;
            var i_global = global.X;
            var j_global = global.Y;

            var a_tile = SharedMemory.Allocate2D<float, Dx>(new(TILE_SIZE, TILE_SIZE), new(TILE_SIZE));
            var b_tile = SharedMemory.Allocate2D<float, Dx>(new(TILE_SIZE, TILE_SIZE), new(TILE_SIZE));

            var sum = 0.0f;

            var i        = Group.IdxX;
            var j        = Group.IdxY;
            var a_i_size = a.IntExtent.X;
            var a_j_size = a.IntExtent.Y;
            var b_i_size = b.IntExtent.X;
            var b_j_size = b.IntExtent.Y;
            for (var k = 0; k < a_i_size; k += TILE_SIZE)
            {
                a_tile[i, j] = i_global < a_i_size && j + k < a_j_size 
                    ? a[i_global, j + k] 
                    : 0;

                b_tile[i, j] = i + k < b_i_size && j_global < b_j_size 
                    ? b[i + k, j_global] 
                    : 0;

                Group.Barrier();

                for (var kk = 0; kk < TILE_SIZE; kk++)
                    sum += a_tile[new(i, kk)] * b_tile[new(kk, j)];

                Group.Barrier();
            }

            if (global.X < result.IntExtent.X && global.Y < result.IntExtent.Y)
                result[global] = sum;
        }

        Console.WriteLine("Copy data to GPU ");
        timer.Restart();

        using var a_buffer      = accelerator.Allocate2DDenseX<float>(new(m, k));
        using var b_buffer      = accelerator.Allocate2DDenseX<float>(new(k, n));
        using var result_buffer = accelerator.Allocate2DDenseX<float>(new(m, n));
        a_buffer.CopyFromCPU(A);
        b_buffer.CopyFromCPU(B);

        Console.WriteLine(timer.Elapsed);

        Console.WriteLine("Begin");
        timer.Restart();

        I2D group_size = new(TILE_SIZE, TILE_SIZE);
        I2D num_groups = new((m + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

        kernel((num_groups, group_size), a_buffer, b_buffer, result_buffer);
        accelerator.Synchronize();

        Console.WriteLine("Computed {0}", timer.Elapsed);

        Console.WriteLine("Copy data from GPU ");
        timer.Restart();

        var actual_result = result_buffer.GetAsArray2D();
        Console.WriteLine(timer.Elapsed);

        Console.WriteLine("      MatrixMultiplyTiledKernel completed");
    }
}
