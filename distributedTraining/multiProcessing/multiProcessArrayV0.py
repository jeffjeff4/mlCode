import multiprocessing


def modify_array(shared_arr, process_id):
    """修改共享数组"""
    for i in range(len(shared_arr)):
        # 通过索引访问和修改数组元素
        shared_arr[i] = shared_arr[i] * 2 + process_id
    print(f"Process {process_id} finished. Array: {list(shared_arr)}")


if __name__ == '__main__':
    # 方法1: 用初始值创建数组
    # shared_array = multiprocessing.Array('i', [1, 2, 3, 4, 5])  # 'i' 表示整数

    # 方法2: 创建指定大小的空数组
    shared_array = multiprocessing.Array('i', 5)  # 创建5个整数的数组
    # 初始化数组值
    for i in range(5):
        shared_array[i] = i + 1

    print(f"Initial array: {list(shared_array)}")

    # 创建多个进程来操作数组
    processes = []
    for i in range(2):
        p = multiprocessing.Process(target=modify_array,
                                    args=(shared_array, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f"Final array: {list(shared_array)}")