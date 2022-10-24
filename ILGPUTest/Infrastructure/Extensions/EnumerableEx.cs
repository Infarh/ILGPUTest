namespace ILGPUTest.Infrastructure.Extensions;

internal static class EnumerableEx
{
    public static void Foreach<T>(this IEnumerable<T> enumerable, Action<T> action)
    {
        foreach (var item in enumerable)
            action(item);
    }
}
