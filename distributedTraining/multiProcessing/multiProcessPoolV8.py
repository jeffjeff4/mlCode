#3. 内存考虑：处理大数据集

import multiprocessing
import itertools


def process_chunk(chunk):
    """处理数据块而不是单个元素"""
    return [x * x for x in chunk]


if __name__ == '__main__':
    # 对于非常大的数据集，可以分块处理
    huge_data = range(1000000)

    # 将数据分成块
    chunk_size = 1000
    chunks = [huge_data[i:i + chunk_size]
              for i in range(0, len(huge_data), chunk_size)]

    with multiprocessing.Pool(4) as pool:
        results = pool.map(process_chunk, chunks)

    # 合并结果
    final_result = list(itertools.chain.from_iterable(results))