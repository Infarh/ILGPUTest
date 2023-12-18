namespace ILGPUTest;

internal static class MatrixMultiplyTest
{
    public static void Run()
    {
        using var context = Context.Create(c => c.Default().EnableAlgorithms());
        using var gpu = context.GetPreferredDevice(false).CreateAccelerator(context);

        float[,] matrix_a =
        {
            {  1,  2,  3 },
            {  4,  5,  6 },
            {  7,  8,  9 },
            { 10, 11, 12 },
        };

        float[,] matrix_b =
        {
            { 13, 14, 15, 16, 17 },
            { 18, 19, 20, 21, 22 },
            { 23, 24, 25, 26, 27 },
        };

        float[,] matrix_c =
        {
            { 118, 124, 130, 136, 142 },
            { 280, 295, 310, 325, 340 },
            { 442, 466, 490, 514, 538 },
            { 604, 637, 670, 703, 736 },
        };


        var cc =  Multiply(matrix_a, matrix_b);
    }

    private static float[,] Multiply(float[,] A, float[,] B)
    {
        using var context = Context.Create(c => c.Default().EnableAlgorithms());
        using var gpu = context.GetPreferredDevice(false).CreateAccelerator(context);

        var kernel = gpu.LoadAutoGroupedStreamKernel((Index2D index, ArrayView2D<float, Stride2D.DenseX> aView, ArrayView2D<float, Stride2D.DenseX> bView, ArrayView2D<float, Stride2D.DenseX> cView) =>
        {
            var x = index.X;
            var y = index.Y;

            var sum = 0f;

            for (var i = 0; i < aView.IntExtent.Y; i++)
                sum += aView[new Index2D(x, i)] * bView[new Index2D(i, y)];

            cView[index] = sum;
        });

        using var a_buffer = gpu.Allocate2DDenseX<float>(new Index2D(A.GetLength(0), A.GetLength(1)));
        using var b_buffer = gpu.Allocate2DDenseX<float>(new Index2D(A.GetLength(1), B.GetLength(1)));
        using var c_buffer = gpu.Allocate2DDenseX<float>(new Index2D(A.GetLength(0), B.GetLength(1)));

        a_buffer.CopyFromCPU(A);
        b_buffer.CopyFromCPU(B);

        kernel(c_buffer.Extent.ToIntIndex(), a_buffer.View, b_buffer.View, c_buffer.View);

        var result = c_buffer.GetAsArray2D();

        return result;
    }
}
