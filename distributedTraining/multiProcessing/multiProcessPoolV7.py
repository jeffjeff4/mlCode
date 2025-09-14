#2. 错误处理

import multiprocessing


def risky_operation(x):
    if x == 13:
        raise ValueError("Unlucky number!")
    return x * 2


if __name__ == '__main__':
    data = [1, 2, 13, 4, 5]

    try:
        with multiprocessing.Pool(2) as pool:
            results = pool.map(risky_operation, data)
        print(results)
    except Exception as e:
        print(f"Error occurred: {e}")