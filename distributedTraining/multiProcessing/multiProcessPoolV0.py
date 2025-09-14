import multiprocessing
import time


def square(x):
    """简单的计算函数"""
    time.sleep(0.1)  # 模拟计算耗时
    return x * x


if __name__ == '__main__':
    # 创建有4个工作进程的池
    with multiprocessing.Pool(processes=4) as pool:
        # 方法1: apply - 同步调用（阻塞）
        result = pool.apply(square, (5,))
        print(f"apply result: {result}")

        # 方法2: apply_async - 异步调用（非阻塞）
        async_result = pool.apply_async(square, (10,))
        print("Doing other work...")
        print(f"async result: {async_result.get()}")  # .get() 会阻塞直到结果就绪

        # 方法3: map - 同步并行映射
        numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        results = pool.map(square, numbers)  # 自动并行计算
        print(f"map results: {results}")

        # 方法4: map_async - 异步并行映射
        async_results = pool.map_async(square, numbers)
        print("Doing other work while computing...")
        results = async_results.get()  # 获取所有结果
        print(f"map_async results: {results}")