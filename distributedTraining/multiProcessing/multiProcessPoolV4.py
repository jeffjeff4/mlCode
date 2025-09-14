import multiprocessing
import os


def process_file(filename):
    """处理单个文件（模拟）"""
    print(f"Processing {filename}")
    # 这里可以是实际的文件处理逻辑，如：
    # with open(filename, 'r') as f:
    #     content = f.read()
    # return len(content)  # 返回文件大小

    # 模拟处理时间
    import time
    time.sleep(1)
    return f"{filename}_processed"


if __name__ == '__main__':
    # 模拟一批要处理的文件
    files = [f"file_{i}.txt" for i in range(12)]

    print(f"Processing {len(files)} files...")

    # 使用进程池并行处理文件
    with multiprocessing.Pool(processes=4) as pool:
        processed_files = pool.map(process_file, files)

    print("\nProcessing completed:")
    for result in processed_files:
        print(f"  {result}")