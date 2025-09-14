import multiprocessing
import os


def process_item(item):
    """处理单个项目"""
    return f"Processed {item} in PID {os.getpid()}"


if __name__ == '__main__':
    data = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    # 使用CPU核心数作为进程数
    num_cores = multiprocessing.cpu_count()
    print(f"Available CPU cores: {num_cores}")

    # 创建进程池，使用所有可用核心
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(process_item, data)

    for result in results:
        print(result)