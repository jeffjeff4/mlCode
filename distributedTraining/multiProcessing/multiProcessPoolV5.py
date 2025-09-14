import multiprocessing
import time


def slow_square(x):
    time.sleep(0.5)
    return x * x


if __name__ == '__main__':
    numbers = [1, 2, 3, 4, 5]

    with multiprocessing.Pool(2) as pool:
        # 1. map: 阻塞式，等待所有结果
        print("Starting map...")
        result_map = pool.map(slow_square, numbers)
        print(f"map results: {result_map}")

        # 2. map_async: 非阻塞，返回AsyncResult对象
        print("Starting map_async...")
        async_result = pool.map_async(slow_square, numbers)
        print("Doing other work while waiting...")
        result_async = async_result.get()  # 这里才阻塞等待结果
        print(f"map_async results: {result_async}")

        # 3. imap: 惰性迭代器，按完成顺序返回结果
        print("Starting imap...")
        for i, result in enumerate(pool.imap(slow_square, numbers)):
            print(f"Result {i}: {result} (received as soon as ready)")