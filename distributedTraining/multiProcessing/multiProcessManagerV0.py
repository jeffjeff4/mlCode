from multiprocessing import Manager


def manager_example():
    with Manager() as manager:
        # Manager 可以创建共享的复杂数据结构
        shared_list = manager.list([1, 2, 3])
        shared_dict = manager.dict({'a': 1, 'b': 2})

        # 但性能比 Value/Array 低，因为需要通过代理