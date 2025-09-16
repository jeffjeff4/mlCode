import multiprocessing as mp

def worker(proc_id, shard, results):
    results[proc_id] = 0

def changeListInfunc(list0):
    len0 = len(list0)
    for idx in range(len0):
        list0[idx] += 10
    return

def main():
    manager = mp.Manager()
    results = manager.list([None])
    p = mp.Process(target=worker, args=(0, [], results))
    p.start()
    p.join()
    results = list(results)
    print(results)

    list0 = [idx for idx in range(10)]
    print("list0 = ", list0)
    changeListInfunc(list0)
    print("list0 = ", list0)

if __name__ == "__main__":
    main()