import multiprocessing


def producer(q, data):
    """生产者进程：向队列放数据"""
    for item in data:
        print(f"Producing {item}")
        q.put(item)  # 放入队列
        # time.sleep(0.1)  # 模拟工作


def consumer(q, name):
    """消费者进程：从队列取数据"""
    while True:
        item = q.get()  # 从队列获取
        if item is None:  # 终止信号
            print(f"Consumer {name} exiting.")
            break
        print(f"Consumer {name} got: {item}")


if __name__ == '__main__':
    # 创建进程安全的队列
    queue = multiprocessing.Queue(maxsize=3)  # 可设置最大容量

    # 创建进程
    p1 = multiprocessing.Process(target=producer, args=(queue, [1, 2, 3, 4, 5]))
    p2 = multiprocessing.Process(target=consumer, args=(queue, "A"))
    p3 = multiprocessing.Process(target=consumer, args=(queue, "B"))

    # 启动消费者
    p2.start()
    p3.start()

    # 启动生产者
    p1.start()

    # 等待生产者结束
    p1.join()

    # 发送终止信号给消费者 (两个消费者，所以发两个None)
    queue.put(None)
    queue.put(None)

    # 等待消费者结束
    p2.join()
    p3.join()

    print("All done!")