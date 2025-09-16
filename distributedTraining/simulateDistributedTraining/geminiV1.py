####this code is runnable, could be used in interview

####simulating distributed ml training with checkpointing
####
####objective:
####implement a simplified version of a distributed machine leanring (ml) training job to demonstrate you understanding of parallelism, data partitioning, and fault tolorance
####
####problem statement:
####you are tasked with simulating a distributed ml training job using parallel trainer processes. the training is simplified to summing numeric data
####
####requirements:
####1. input data
####using a csv file or an im-memory array that contains one numeric value per row(re.g., [1,2,3,..., 1000])
####
####2. parallel trainers:
####launch N parallel "trainer" processes
####each trainer should be load and process a distinct shard of the data (non-overlapping rows)
####
####3. simulated training
####each trainer computes the sum of its assigned shard
####then combine the partial results to compute the global sum
####
####4. sum of the number at every straining step
####
####5. load the data in random order
####
####6. make the order (still random) deterministic between 2 training runs
####
####7. please use this code as an example
####please refer to geminiV0.py


import multiprocessing as mp
import os
import time
import sys
import random
import numpy as np

# Define a directory for storing checkpoints
CHECKPOINT_DIR = "./checkpoints"


def worker(proc_id, shard, batch_size, queue_in, queue_out, simulate_failure_flag=False):
    """
    工作進程，處理其資料分片並在每個 mini-batch 後同步總和。
    """
    partial_sum = 0
    start_batch_idx = 0
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_{proc_id}.txt")

    try:
        # 從檢查點恢復進度
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                data = f.read().split(',')
                partial_sum = float(data[0])
                start_batch_idx = int(data[1])
            print(
                f"進程 {proc_id}: 從檢查點恢復。部分總和: {partial_sum}, 從批次 {start_batch_idx} 開始")
        else:
            print(f"進程 {proc_id}: 開始新的訓練任務。")

        # 按 mini-batch 處理資料分片
        for i in range(start_batch_idx, (len(shard) + batch_size - 1) // batch_size):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(shard))
            mini_batch = shard[start:end]

            # 每個訓練步驟的本地求和
            local_batch_sum = sum(mini_batch)

            # 模擬故障
            if simulate_failure_flag and i == (len(shard) // batch_size) // 2:
                print(f"進程 {proc_id}: 模擬故障並退出。")
                sys.exit(1)

            # 將本地結果傳送給主進程
            queue_out.put(local_batch_sum)
            # 從主進程接收全局總和
            global_sum_from_main = queue_in.get()

            # 更新部分總和
            partial_sum += global_sum_from_main

            # 定期儲存檢查點
            if (i + 1) % 5 == 0:
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                with open(checkpoint_file, 'w') as f:
                    f.write(f"{partial_sum},{i + 1}")
                print(f"進程 {proc_id}: 在步驟 {i + 1} 儲存檢查點。")

        print(f"進程 {proc_id}: 完成。最終總和為 {partial_sum}。")
        # 將最終結果傳回
        queue_out.put(partial_sum)

    except Exception as e:
        print(f"進程 {proc_id} 發生錯誤: {e}")
        # 結果將不會被寫入，在共享列表中留下 None


def create_data_shards(data, num_processes):
    """
    將資料分割成 N 個不重疊的分片。
    """
    avg_shard_size = len(data) // num_processes
    shards = []
    for i in range(num_processes):
        start = i * avg_shard_size
        end = (i + 1) * avg_shard_size if i < num_processes - 1 else len(data)
        shards.append(data[start:end])
    return shards


def clean_checkpoints():
    """
    移除所有檢查點檔案。
    """
    print("正在清理舊的檢查點...")
    if os.path.exists(CHECKPOINT_DIR):
        for f in os.listdir(CHECKPOINT_DIR):
            os.remove(os.path.join(CHECKPOINT_DIR, f))
        os.rmdir(CHECKPOINT_DIR)
    print("檢查點已清理。")


def run_simulation(data, num_processes, batch_size, simulate_failure=False):
    """
    執行分散式訓練模擬的一個回合。
    """
    if not simulate_failure:
        print("\n--- 正在執行模擬 ---")

    shards = create_data_shards(data, num_processes)
    num_batches = (len(shards[0]) + batch_size - 1) // batch_size

    # 為每個進程創建輸入和輸出佇列
    queue_ins = [mp.Queue() for _ in range(num_processes)]
    queue_outs = [mp.Queue() for _ in range(num_processes)]

    processes = []
    for i in range(num_processes):
        p_args = (i, shards[i], batch_size, queue_ins[i], queue_outs[i])
        if simulate_failure and i == 1:
            p = mp.Process(target=worker, args=p_args + (True,))
        else:
            p = mp.Process(target=worker, args=p_args)
        processes.append(p)
        p.start()

    # 主進程的 mini-batch 迴圈
    for _ in range(num_batches):
        partial_sums = []
        # 從每個進程收集本地總和
        for q_out in queue_outs:
            partial_sum = q_out.get()
            partial_sums.append(partial_sum)

        # 模擬 all_reduce: 求和並廣播
        global_sum = sum(partial_sums)
        for q_in in queue_ins:
            q_in.put(global_sum)

    final_sums = []
    incomplete_run = False
    for q_out in queue_outs:
        final_sum = q_out.get()
        if final_sum is not None:
            final_sums.append(final_sum)
        else:
            incomplete_run = True

    final_result = sum(final_sums)

    print("\n--- 彙總結果 ---")
    print(f"計算總和: {final_result}")
    return final_result, incomplete_run, processes


def main():
    """
    主函數，編排模擬並執行測試。
    """
    DATA_SIZE = 1000
    BATCH_SIZE = 100
    NUM_PROCESSES = 4

    # 確保隨機順序在多次運行中是確定的
    random.seed(42)
    data = list(range(1, DATA_SIZE + 1))
    random.shuffle(data)

    EXPECTED_SUM = sum(data)

    # --- 案例 1: 正常執行 ---
    clean_checkpoints()
    final_sum, _, processes = run_simulation(data, NUM_PROCESSES, BATCH_SIZE, simulate_failure=False)
    for p in processes: p.join()
    print(f"預期總和: {EXPECTED_SUM}")
    print(f"正常情況通過: {final_sum == EXPECTED_SUM}")

    # --- 案例 2: 故障與恢復 ---
    print("\n\n--- 執行故障與恢復測試 ---")
    print("--- 第一次運行: 模擬進程故障 ---")
    clean_checkpoints()
    final_sum, incomplete_run, processes = run_simulation(data, NUM_PROCESSES, BATCH_SIZE, simulate_failure=True)
    for p in processes: p.join()

    if incomplete_run:
        print("\n偵測到故障。第一次運行因模擬崩潰而不完整。")
        print("發起新的運行，從上一個檢查點恢復。")

        print("\n--- 第二次運行: 恢復 ---")
        final_sum, _, processes = run_simulation(data, NUM_PROCESSES, BATCH_SIZE, simulate_failure=False)
        for p in processes: p.join()

        print("\n--- 恢復測試結果 ---")
        print(f"預期總和: {EXPECTED_SUM}")
        print(f"恢復後的最終總和: {final_sum}")
        print(f"恢復案例通過: {final_sum == EXPECTED_SUM}")
    else:
        print("錯誤: 故障模擬行為未如預期。")


if __name__ == "__main__":
    mp.freeze_support()
    main()
