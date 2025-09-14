import multiprocessing


def shared_memory_worker(counter, arr):
    """操作共享内存的进程"""
    with counter.get_lock():  # 必须加锁！
        counter.value += 1

    # 修改共享数组
    for i in range(len(arr)):
        arr[i] *= 2  # 每个元素乘以2


if __name__ == '__main__':
    # 创建共享值 (typecode: 'i' for int, 'd' for double)
    counter = multiprocessing.Value('i', 0)

    # 创建共享数组
    arr = multiprocessing.Array('i', [1, 2, 3, 4, 5])  # 'i' 表示整数

    processes = []
    for i in range(3):
        p = multiprocessing.Process(target=shared_memory_worker,
                                    args=(counter, arr))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f"Final counter value: {counter.value}")  # 输出: 3
    print(f"Final array: {list(arr)}")  # 输出: [8, 16, 24, 32, 40] (被多个进程多次x2)