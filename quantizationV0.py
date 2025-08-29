####请举例说明量化的end to end 流程，对Q， K， V的处理，何时量化，何时反量化，量化如何影响矩阵乘等等
####
####好的，我们来详细梳理一个端到端（end-to-end）的量化流程，重点关注 Transformer 模型中 Q、K、V 的处理，以及量化如何影响矩阵乘法。
####
####我们将以一个简单的前向传播为例，假设我们对 Q、K、V 的计算以及随后的点积进行量化。
####
####1. 为什么需要量化？
####在高精度训练（如 FP32）后，模型的权重和激活值都是 32 位浮点数。量化的目标是将这些高精度值转换为低精度整数（如 INT8），以减少模型大小、加速计算并降低功耗。
####
####2. 量化流程概览
####一个典型的量化流程分为三步：
####
####量化（Quantization）：将 FP32 值映射到 INT8 整数。
####
####量化计算（Quantized Operations）：在 INT8 域内执行矩阵乘法、加法等运算。
####
####反量化（Dequantization）：将 INT8 结果映射回 FP32，用于下游非量化运算（如 Softmax）。
####
####3. 端到端量化流程示例
####让我们以 Transformer 的注意力机制为例，详细分解 Q、K、V 的量化过程。
####
####步骤 1：获取 Q、K、V 张量
####假设我们已经训练好了一个 FP32 的 Transformer 模型。现在，一个输入张量 X 通过 Q、K、V 的线性层，得到 FP32 的 Q、K、V 张量。
####
####Q (FP32)：[batch_size, seq_len, head_dim]
####
####K (FP32)：[batch_size, seq_len, head_dim]
####
####V (FP32)：[batch_size, seq_len, head_dim]
####
####步骤 2：对 Q、K 进行量化
####在进行点积之前，我们需要对 Q 和 K 进行量化。这里我们使用 per-tensor 量化为例。
####
####量化参数：scale 和 zero_point。这些参数通常在校准阶段（Calibration）计算，通过分析大量样本的激活值分布来确定。
####
####量化操作：
####
####量化 Q：Q_int8 = round(Q / scale_q + zero_point_q)
####
####量化 K：K_int8 = round(K / scale_k + zero_point_k)
####
####步骤 3：量化矩阵乘法
####现在我们对量化后的 Q 和 K 进行点积。
####
####输入：Q_int8 和 K_int8，它们是 INT8 整数张量。
####
####计算：
####scores_int32 = Q_int8 * K_int8^T
####
####量化如何影响矩阵乘法？
####量化矩阵乘法是用整数运算模拟浮点运算。它会产生一个更高精度的整数结果（通常是 INT32），以避免在中间过程产生误差。这个 scores_int32 结果不是 FP32，而是一个中间的整数值。
####
####为了得到这个结果的真实值，我们需要将它乘以量化参数：
####scores_fp32 = (scores_int32 - Z_s) * S_s
####其中 S_s 和 Z_s 是量化参数。这个公式可以简化成：
####scores_fp32 = (Q_int8 * K_int8^T) * (scale_q * scale_k) - (something)
####
####这个过程通常是在硬件层面完成的，例如英伟达的 Tensor Core 可以高效地执行 INT8 矩阵乘法。
####
####步骤 4：反量化 Softmax 输入
####Transformer 中的 Softmax 模块通常不进行量化，因为它涉及指数运算，这在整数域中难以高效实现。
####
####因此，我们需要将量化点积的结果 scores_int32 反量化回 FP32，然后将其输入 Softmax。
####
####反量化操作：
####scores_fp32 = scores_int32 * (scale_q * scale_k) + ...
####
####Softmax：attention_weights = softmax(scores_fp32)
####
####步骤 5：对 V 张量进行量化
####在计算注意力输出之前，我们还需要量化 V 张量。
####
####量化 V：V_int8 = round(V / scale_v + zero_point_v)
####
####步骤 6：量化矩阵乘法并反量化
####最后一步是将注意力权重 attention_weights 与 V_int8 进行点积。
####
####输入：attention_weights (FP32) 和 V_int8 (INT8)。
####
####计算：
####output_int32 = attention_weights_fp32 * V_int8
####这个乘法通常是混合精度运算。
####
####反量化：
####output_fp32 = output_int32 * scale_v + ...
####
####总结
####这个端到端的流程展示了量化并非简单地将所有 FP32 值转换为 INT8。关键在于：
####
####量化与反量化的时机：只对矩阵乘法的输入（Q、K、V）进行量化，并在非整数运算（如 Softmax）之前进行反量化。
####
####量化计算：矩阵乘法在量化后的 INT8 域内完成，中间结果通常是更高精度的整数（INT32），以保持精度。
####
####量化对矩阵乘法的影响：量化将浮点乘法转换为整数乘法，大大加速了计算，但需要处理量化参数和零点，以确保结果的正确性。
####
####这个流程是现代量化技术的基础，是模型在部署到边缘设备或专用硬件时实现高性能的关键。


import torch
import torch.nn as nn
import torch.quantization
import torch.nn.functional as F

import torch
import torch.nn as nn


def compare_model_weights(model_a, model_b):
    models_differ = 0
    for key_a, key_b in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        if key_a[0] != key_b[0]:
            print(f"Mismatch in key names: {key_a[0]} vs {key_b[0]}")
            models_differ += 1
            continue
        if not torch.equal(key_a[1], key_b[1]):
            print(f"Mismatch found at key: {key_a[0]}")
            models_differ += 1
    if models_differ == 0:
        print("Models have identical weights.")
    else:
        print(f"Models differ in {models_differ} parameter sets.")


# Example usage:
# model1 = MyModel()
# model2 = MyModel()
# compare_model_weights(model1, model2)


# 1. 定义一个简单的模型，包含一个模拟的 QKV 线性层
class AttentionModel(nn.Module):
    def __init__(self, in_dim, head_dim):
        super().__init__()
        self.q_proj = nn.Linear(in_dim, head_dim)
        self.k_proj = nn.Linear(in_dim, head_dim)
        self.v_proj = nn.Linear(in_dim, head_dim)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 模拟注意力点积
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Softmax 不量化，在 FP32 域计算
        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, v)
        return output


# 2. 准备模型和量化工具
# 设置为评估模式，因为量化通常在推理时进行
model = AttentionModel(in_dim=32, head_dim=16).eval()

# 创建一个 QuantStub 和 DeQuantStub
# QuantStub: 插入量化操作，将 FP32 -> INT8
# DeQuantStub: 插入反量化操作，将 INT8 -> FP32
quant_stub = torch.quantization.QuantStub()
dequant_stub = torch.quantization.DeQuantStub()


# 3. 模拟端到端的量化前向传播

def quantized_attention_forward(model, x, quant_stub, dequant_stub):
    # 输入的 Q、K、V 张量
    q_fp32 = model.q_proj(x)
    k_fp32 = model.k_proj(x)
    v_fp32 = model.v_proj(x)

    # 3.1. 对 Q 和 K 进行量化（per-tensor）
    # 在这个阶段，PyTorch会根据校准数据自动决定 scale 和 zero_point
    q_quant = quant_stub(q_fp32)
    k_quant = quant_stub(k_fp32)

    # 3.2. 执行量化矩阵乘法（在整数域）
    # PyTorch的量化API会在后台处理这个过程，模拟整数乘法
    # 结果是一个中间的整数张量
    scores_quant = torch.matmul(q_quant, k_quant.transpose(-2, -1))

    # 3.3. 反量化，为 Softmax 做准备
    # Softmax 无法在整数域高效实现，所以需要反量化回 FP32
    scores_fp32 = dequant_stub(scores_quant)

    # 3.4. Softmax 在 FP32 域计算
    attention_weights = F.softmax(scores_fp32, dim=-1)

    # 3.5. 对 V 进行量化
    # 另一个量化操作，通常与 Q、K 的量化分开进行
    v_quant = quant_stub(v_fp32)

    # 3.6. 混合精度矩阵乘法：FP32 * INT8
    # 这里的 attention_weights 是 FP32
    output_quant = torch.matmul(attention_weights, v_quant)

    # 3.7. 最终反量化得到 FP32 输出
    final_output = dequant_stub(output_quant)

    return final_output


# 4. 运行示例
# 假定输入张量
x = torch.randn(4, 8, 32)  # batch_size=4, seq_len=8, in_dim=32

# 为了进行量化，我们需要先进行一次"校准"（calibration）
# 这一步是为了让 PyTorch 收集激活值的分布信息，以便计算出合适的量化参数
print("--- 1. 校准阶段 ---")
model.q_proj = nn.Sequential(quant_stub, model.q_proj, dequant_stub)
model.k_proj = nn.Sequential(quant_stub, model.k_proj, dequant_stub)
model.eval()
torch.quantization.prepare(model, inplace=True)
with torch.no_grad():
    model(x)  # 运行一次前向传播，收集量化数据
print("校准完成，量化参数已确定")

# 现在我们将模型转换为量化版本
model.q_proj = model.q_proj[1]  # 移除暂时的量化/反量化包装
model.k_proj = model.k_proj[1]
model.add_module('quant_stub', quant_stub)
model.add_module('dequant_stub', dequant_stub)

# 转换模型为量化模型
quantized_model = torch.quantization.convert(model, inplace=True)
print("\n--- 2. 模型转换完成 ---")
print(quantized_model)

compare_model_weights(model, quantized_model)

# 5. 使用量化模型进行推理
print("\n--- 3. 量化前向传播（推理） ---")
with torch.no_grad():
    # 运行我们模拟的量化前向传播
    output = quantized_attention_forward(quantized_model, x, quant_stub, dequant_stub)
    print("输出张量的形状:", output.shape)
    print("输出张量的数据类型:", output.dtype)