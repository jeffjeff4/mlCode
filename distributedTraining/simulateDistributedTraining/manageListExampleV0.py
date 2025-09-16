import multiprocessing as mp
import time


def worker(process_id, data, shared_results):
    """工作进程函数"""
    print(f"Process {process_id} started with data: {data}")

    # 模拟一些工作
    time.sleep(1)
    result = f"Result from process {process_id}: {len(data)} items processed"

    # 将结果放入共享列表
    shared_results[process_id] = result
    print(f"Process {process_id} finished. Shared results: {list(shared_results)}")


def basic_example():
    """基础示例：使用 Manager.list() 共享数据"""
    # 创建进程管理器
    manager = mp.Manager()

    # 创建共享列表，初始化为 [None, None, None]
    num_processes = 3
    shared_results = manager.list([None] * num_processes)

    print(f"Initial shared results: {list(shared_results)}")

    # 创建并启动进程
    processes = []
    for i in range(num_processes):
        p = mp.Process(
            target=worker,
            args=(i, [f"item_{i}_{j}" for j in range(i + 1)], shared_results)
        )
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    print(f"Final shared results: {list(shared_results)}")
    return list(shared_results)


if __name__ == "__main__":
    results = basic_example()