import multiprocessing as mp

def worker(proc_id, shard, results):
    results[proc_id] = 0

def main():
    manager = mp.Manager()
    results = manager.list([None])
    p = mp.Process(target=worker, args=(0, [], results))
    p.start()
    p.join()
    results = list(results)
    print(results)

if __name__ == "__main__":
    main()