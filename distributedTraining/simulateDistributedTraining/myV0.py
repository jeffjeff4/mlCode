import multiprocessing as mp

# you can write to stdout for debugging purposes, e.g.
print("This is a debug message")

## 10 proccesses
##

DATA_SIZE = 1000
NUM_PROC = 4


# geberate training data for each worker
def createDataPart(data, num_proc):
    average_len = len(data) // num_proc
    data_parts = []
    for idx in range(num_proc):
        start = idx * average_len
        end = (idx + 1) * average_len
        if idx == num_proc - 1:
            end = len(data)

        tup = data[start:end]
        data_parts.append(tup)

    return data_parts


# each worker logic
def worker(proc_id, data_part, results):
    local_sum = 0
    start = 0

    len0 = len(data_part)
    for idx in range(len0):
        local_sum += data_part[idx]

    results[proc_id] = local_sum
    print("proc_id = ", proc_id, ", local_sum = ", local_sum)
    return


# simulate the parallel training
def simulation(data, num_proc):
    data_parts = createDataPart(data, num_proc)

    manager = mp.Manager()
    results = manager.list([None] * num_proc)
    procs = []

    for idx in range(num_proc):
        p_arg = (idx, data_parts[idx], results)

        p = mp.Process(target=worker, args=p_arg)
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    rst = 0
    for tmp in results:
        if tmp != None:
            rst += tmp

    results = list(results)
    print(results)

    print("rst = ", rst)
    return rst


def main():
    data = list(range(1, 1 + DATA_SIZE))
    expected_rst = sum(data)

    final_sum = simulation(data, NUM_PROC)
    print("expected_rst = ", expected_rst, ", final_sum = ", final_sum)


if __name__ == "__main__":
    main()

# import multiprocessing as mp


# def worker(proc_id, shard, results):
#     results[proc_id] = 0


# def main():
#     manager = mp.Manager()
#     results = manager.list([None])
#     p = mp.Process(target=worker, args=(0, [], results))
#     p.start()
#     p.join()
#     results = list(results)
#     print(results)


