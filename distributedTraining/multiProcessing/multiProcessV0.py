import multiprocessing
import time
import os

def worker(name, seconds):
    """子进程要执行的任务"""
    print(f"Process {name} (PID: {os.getpid()}) started. Sleeping for {seconds} seconds")
    time.sleep(seconds)
    print(f"Process {name} finished.")

if __name__ == '__main__':  # 必须的！用于Windows和类Unix系统的兼容性
    # 创建进程对象
    p1 = multiprocessing.Process(target=worker, args=("Alice", 2))
    p2 = multiprocessing.Process(target=worker, args=("Bob", 3))

    # 启动进程
    p1.start()
    p2.start()

    print("Main process waiting for subprocesses...")

    # 等待进程结束
    p1.join()
    p2.join()

    print("All processes completed!")