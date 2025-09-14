import multiprocessing


def process_with_params(x, multiplier, power):
    """带多个参数的函数"""
    return (x * multiplier) ** power


if __name__ == '__main__':
    numbers = [1, 2, 3, 4, 5]

    # 方法1: 使用lambda函数（注意：有些环境可能不支持）
    with multiprocessing.Pool(4) as pool:
        results = pool.map(lambda x: process_with_params(x, 10, 2), numbers)
    print(f"Lambda results: {results}")


    # 方法2: 使用局部函数（推荐）
    def wrapper_func(x):
        return process_with_params(x, multiplier=10, power=2)


    with multiprocessing.Pool(4) as pool:
        results = pool.map(wrapper_func, numbers)
    print(f"Wrapper results: {results}")

    # 方法3: 使用functools.partial（最佳方式）
    from functools import partial

    processed_func = partial(process_with_params, multiplier=10, power=2)

    with multiprocessing.Pool(4) as pool:
        results = pool.map(processed_func, numbers)
    print(f"Partial results: {results}")