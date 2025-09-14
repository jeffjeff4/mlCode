####总结
####pool.map() 的主要特点：
####
####特性	描述
####易用性	和内置 map() 类似的接口
####自动化	自动处理进程创建、任务分配、结果收集
####顺序保持	返回结果的顺序与输入顺序一致
####阻塞操作	等待所有任务完成才返回
####性能提升	对CPU密集型任务效果显著
####最佳实践：
####
####用于CPU密集型任务，而不是I/O密集型
####
####使用 with 语句自动管理池生命周期
####
####对于需要参数的函数，使用 functools.partial
####
####处理大量数据时考虑合适的 chunksize
####
####在 if __name__ == '__main__': 块中运行

#1. 性能优化：块大小（chunksize）

import multiprocessing


def process_item(x):
    return x * x


if __name__ == '__main__':
    large_data = list(range(1000))

    with multiprocessing.Pool(4) as pool:
        # 默认块大小（可能不是最优）
        results1 = pool.map(process_item, large_data)

        # 手动指定块大小（减少进程间通信开销）
        results2 = pool.map(process_item, large_data, chunksize=25)