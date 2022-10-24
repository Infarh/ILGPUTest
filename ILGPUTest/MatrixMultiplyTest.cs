using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ILGPUTest;

public static class MatrixMultiplyTest
{
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

        var m = A.GetLength(0);
        var k = A.GetLength(1);
        var n = B.GetLength(1);

        using var context     = Context.CreateDefault();
        using var accelerator = context.CreateCudaAccelerator(0);

        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>>(
            MatrixMultiplyAcceleratedKernel);

        static void MatrixMultiplyAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> AView,
            ArrayView2D<float, Stride2D.DenseX> BView,
            ArrayView2D<float, Stride2D.DenseX> ResultView)
        {
            var x   = index.X;
            var y   = index.Y;
            var sum = 0.0f;

            for (var i = 0; i < AView.IntExtent.Y; i++)
                sum += AView[new(x, i)] * BView[new(i, y)];

            ResultView[index] = sum;
        }

        using var a_buffer      = accelerator.Allocate2DDenseX<float>(new(m, k));
        using var b_buffer      = accelerator.Allocate2DDenseX<float>(new(k, n));
        using var result_buffer = accelerator.Allocate2DDenseX<float>(new(m, n));
        a_buffer.CopyFromCPU(A);
        b_buffer.CopyFromCPU(B);

        kernel(result_buffer.Extent.ToIntIndex(), a_buffer.View, b_buffer.View, result_buffer.View);

        var actual_c = result_buffer.GetAsArray2D();

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

        var m = A.GetLength(0);
        var k = A.GetLength(1);
        var n = B.GetLength(1);

        using var context = Context.CreateDefault();
        using var accelerator = context.CreateCudaAccelerator(0);

        var kernel = accelerator.LoadStreamKernel<
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>>(
            MatrixMultiplyTiledKernel);

        const int TILE_SIZE = 2;

        static void MatrixMultiplyTiledKernel(
            ArrayView2D<float, Stride2D.DenseX> AView,
            ArrayView2D<float, Stride2D.DenseX> BView,
            ArrayView2D<float, Stride2D.DenseX> ResultView)
        {
            var global = Grid.GlobalIndex.XY;
            var x      = Group.IdxX;
            var y      = Group.IdxY;

            var a_tile = SharedMemory.Allocate2D<float, Stride2D.DenseX>(new(TILE_SIZE, TILE_SIZE), new(TILE_SIZE));
            var b_tile = SharedMemory.Allocate2D<float, Stride2D.DenseX>(new(TILE_SIZE, TILE_SIZE), new(TILE_SIZE));
            var sum    = 0.0f;

            for (var i = 0; i < AView.IntExtent.X; i += TILE_SIZE)
            {
                if (global.X < AView.IntExtent.X && y + i < AView.IntExtent.Y)
                    a_tile[x, y] = AView[global.X, y + i];
                else
                    a_tile[x, y] = 0;

                if (x + i < BView.IntExtent.X && global.Y < BView.IntExtent.Y)
                    b_tile[x, y] = BView[x + i, global.Y];
                else
                    b_tile[x, y] = 0;

                Group.Barrier();

                for (var k = 0; k < TILE_SIZE; k++)
                    sum += a_tile[new(x, k)] * b_tile[new(k, y)];

                Group.Barrier();
            }

            if (global.X < ResultView.IntExtent.X && global.Y < ResultView.IntExtent.Y)
                ResultView[global] = sum;
        }

        var group_size = new Index2D(TILE_SIZE, TILE_SIZE);
        var num_groups = new Index2D((m + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

        using var a_buffer      = accelerator.Allocate2DDenseX<float>(new(m, k));
        using var b_buffer      = accelerator.Allocate2DDenseX<float>(new(k, n));
        using var result_buffer = accelerator.Allocate2DDenseX<float>(new(m, n));
        a_buffer.CopyFromCPU(A);
        b_buffer.CopyFromCPU(B);

        kernel((num_groups, group_size), a_buffer, b_buffer, result_buffer);

        var actual_result = result_buffer.GetAsArray2D();
    }
}
