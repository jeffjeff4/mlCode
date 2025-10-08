import numpy as np
import pandas as pd
import math
from io import StringIO


# --- 餘弦退火學習率排程器模擬 ---

def cosine_annealing_lr(t_cur, t_max, lr_min, lr_max):
    """
    計算餘弦退火排程下的當前學習率 (LR_t)。

    參數:
    t_cur (int): 當前步數 (例如: 當前 Epoch 數)。
    t_max (int): 總步數 (例如: 總 Epoch 數)。
    lr_min (float): 學習率最小值 (LR_min)。
    lr_max (float): 學習率最大值 (LR_max)。

    回傳:
    float: 當前計算出的學習率。
    """
    # 確保 t_cur 不超過 t_max，避免餘弦函數值出錯
    t_cur = np.clip(t_cur, 0, t_max)

    # 餘弦退火公式:
    # LR_t = LR_min + 0.5 * (LR_max - LR_min) * (1 + cos(T_cur/T_max * pi))
    lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos((t_cur / t_max) * math.pi))

    return lr_t


if __name__ == '__main__':
    # --- 參數設定 ---
    LR_MAX = 1e-3  # 初始最大學習率 (LR_max)
    LR_MIN = 1e-5  # 學習率最小值 (LR_min)
    T_MAX = 20  # 總訓練週期 (T_max, 這裡設定為 20 個 Epoch)

    print(f"--- 餘弦退火排程模擬 (總 Epoch: {T_MAX}) ---")
    print(f"初始學習率 (LR_MAX): {LR_MAX}")
    print(f"最小學習率 (LR_MIN): {LR_MIN}\n")

    results = []

    # 模擬從 Epoch 0 到 Epoch T_MAX 的訓練過程
    for epoch in range(T_MAX + 1):
        # T_cur = epoch
        current_lr = cosine_annealing_lr(epoch, T_MAX, LR_MIN, LR_MAX)

        results.append({
            'Epoch': epoch,
            'Learning Rate': f'{current_lr:.8f}'
        })

    # 將結果轉換為 DataFrame 以便清晰展示
    df = pd.DataFrame(results)

    # 使用 StringIO 模擬輸出為 Markdown 表格
    output = StringIO()
    df.to_markdown(output, index=False)

    print(output.getvalue())
    print("\n--- 觀察趨勢 ---")
    print("學習率一開始下降緩慢，中期加速下降，最後接近 LR_MIN 時又趨於平緩。")
