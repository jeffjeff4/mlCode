import multiprocessing
import time


def square(x):
    """计算平方的函数"""
    print(f"Processing {x} in process {multiprocessing.current_process().name}")
    time.sleep(0.5)  # 模拟计算耗时
    return x * x


if __name__ == '__main__':
    # 输入数据
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    print("Sequential processing (slow):")
    start_time = time.time()
    results_seq = [square(x) for x in numbers]  # 顺序执行
    print(f"Time: {time.time() - start_time:.2f}s")
    print(f"Results: {results_seq}")

    print("\nParallel processing with pool.map (fast):")
    start_time = time.time()

    # 创建有4个工作进程的池
    with multiprocessing.Pool(processes=4) as pool:
        # 使用 pool.map 并行计算
        results_parallel = pool.map(square, numbers)

    print(f"Time: {time.time() - start_time:.2f}s")
    print(f"Results: {results_parallel}")