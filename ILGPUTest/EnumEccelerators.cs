namespace ILGPUTest;

public static class EnumAcceleratorsTest
{
    public static void Run()
    {
        using var context = Context.Create(builder => builder.AllAccelerators());

        var i = 1;
        foreach (var device in context)
        {
            Console.WriteLine("{0}.", i++);
            Console.WriteLine(device);
        }

    }
}
