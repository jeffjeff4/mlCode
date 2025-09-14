import multiprocessing


def pipe_worker(conn, name):
    """使用管道通信的进程"""
    # 从管道接收消息
    message = conn.recv()
    print(f"{name} received: {message}")

    # 通过管道发送响应
    conn.send(f"Hello from {name}!")
    conn.close()  # 关闭连接


if __name__ == '__main__':
    # 创建管道，返回两个连接对象
    parent_conn, child_conn = multiprocessing.Pipe()

    p = multiprocessing.Process(target=pipe_worker,
                                args=(child_conn, "ChildProcess"))
    p.start()

    # 主进程通过管道发送消息
    parent_conn.send("Message from main process")

    # 接收子进程的响应
    response = parent_conn.recv()
    print(f"Main process got: {response}")

    p.join()