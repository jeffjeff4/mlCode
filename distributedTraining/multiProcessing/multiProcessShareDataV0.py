import multiprocessing
import ctypes  # 用于更复杂的类型定义


class SharedData:
    """共享数据结构"""

    def __init__(self):
        # 共享计数器
        self.counter = multiprocessing.Value('i', 0)
        # 共享标志
        self.flag = multiprocessing.Value('b', False)  # 布尔值
        # 共享数组
        self.data_array = multiprocessing.Array('d', 10)  # 10个双精度浮点数

    def display(self):
        """显示当前状态"""
        print(f"Counter: {self.counter.value}, Flag: {self.flag.value}")
        print(f"Array: {list(self.data_array)}")


def complex_worker(shared_data, worker_id):
    """操作复杂共享数据的进程"""
    with shared_data.counter.get_lock():
        shared_data.counter.value += worker_id + 1

    # 修改数组
    for i in range(len(shared_data.data_array)):
        shared_data.data_array[i] = worker_id * 10 + i

    # 设置标志
    shared_data.flag.value = True

    print(f"Worker {worker_id} completed")


if __name__ == '__main__':
    # 创建共享数据结构
    shared_data = SharedData()

    print("Before processing:")
    shared_data.display()

    # 创建并启动进程
    processes = []
    for i in range(3):
        p = multiprocessing.Process(target=complex_worker,
                                    args=(shared_data, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("\nAfter processing:")
    shared_data.display()