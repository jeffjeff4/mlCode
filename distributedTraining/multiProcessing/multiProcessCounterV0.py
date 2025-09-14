import multiprocessing
import time


def increment_counter(counter, process_name):
    """增加共享计数器的值"""
    for _ in range(5):
        time.sleep(0.1)  # 模拟一些工作

        # 必须使用锁来安全地修改值！
        with counter.get_lock():
            counter.value += 1
            print(f"{process_name}: Counter = {counter.value}")


if __name__ == '__main__':
    # 创建共享整数值，初始值为0
    counter = multiprocessing.Value('i', 0)  # 'i' 表示整数

    # 创建多个进程来操作同一个计数器
    processes = []
    for i in range(3):
        p = multiprocessing.Process(target=increment_counter,
                                    args=(counter, f"Process-{i + 1}"))
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    print(f"Final counter value: {counter.value}")


####1. 必须使用锁！
##### 错误方式（可能导致竞争条件）
##### counter.value += 1
####
##### 正确方式
####with counter.get_lock():
####    counter.value += 1
####
####2. 性能考虑
##### 不好：频繁的小操作都加锁
####for i in range(1000):
####    with counter.get_lock():
####        counter.value += 1
####
##### 更好：批量操作后一次更新
####local_sum = 0
####for i in range(1000):
####    local_sum += 1
####with counter.get_lock():
####    counter.value += local_sum
####
####
####3. 类型安全
##### 确保使用正确的类型代码
####counter = multiprocessing.Value('i', 0)  # 整数
####temperature = multiprocessing.Value('d', 25.5)  # 双精度浮点数
####
####
####4. 替代方案：Manager
####对于更复杂的数据结构，可以使用 multiprocessing.Manager：
####from multiprocessing import Manager
####
####
####def manager_example():
####    with Manager() as manager:
####        # Manager 可以创建共享的复杂数据结构
####        shared_list = manager.list([1, 2, 3])
####        shared_dict = manager.dict({'a': 1, 'b': 2})
####
####        # 但性能比 Value/Array 低，因为需要通过代理