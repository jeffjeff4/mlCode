##LSTM
##graph LR
##    x[输入x] --> 合并
##    h[隐状态h] --> 合并
##    合并 --> 输入门
##    合并 --> 遗忘门
##    合并 --> 输出门
##    合并 --> 候选记忆
##    输入门 --> 记忆更新
##    遗忘门 --> 记忆更新
##    候选记忆 --> 记忆更新
##    记忆更新 --> 新记忆c
##    新记忆c --> 新隐状态h
##    输出门 --> 新隐状态h
##
##GRU
##graph LR
##    x[输入x] --> 合并
##    h[隐状态h] --> 合并
##    合并 --> 更新门
##    合并 --> 重置门
##    重置门 --> 候选隐状态
##    x --> 候选隐状态
##    更新门 --> 输出h
##    候选隐状态 --> 输出h
##
##
##AUGRU
### 与传统GRU的唯一区别：
##attn_weight = sigmoid(attention_layer(h_tilde))  # 计算注意力权重
##z = attn_weight * z  # 动态调整更新门


import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 输入门、遗忘门、输出门、候选记忆
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.candidate_mem = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, prev_h, prev_c):
        combined = torch.cat([x, prev_h], dim=-1)
        # 门控计算
        i = torch.sigmoid(self.input_gate(combined))    # 输入门
        f = torch.sigmoid(self.forget_gate(combined))   # 遗忘门
        o = torch.sigmoid(self.output_gate(combined))   # 输出门
        c_tilde = torch.tanh(self.candidate_mem(combined))  # 候选记忆
        # 状态更新
        c = f * prev_c + i * c_tilde                  # 记忆单元更新
        h = o * torch.tanh(c)                         # 隐状态更新
        return h, c


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 更新门、重置门、候选隐状态
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, prev_h):
        combined = torch.cat([x, prev_h], dim=-1)
        # 门控计算
        z = torch.sigmoid(self.update_gate(combined))  # 更新门
        r = torch.sigmoid(self.reset_gate(combined))   # 重置门
        combined_reset = torch.cat([x, r * prev_h], dim=-1)
        h_tilde = torch.tanh(self.candidate(combined_reset))  # 候选隐状态
        # 状态更新
        h = (1 - z) * prev_h + z * h_tilde
        return h


class AUGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # GRU基础门控
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        # 注意力层
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, prev_h):
        combined = torch.cat([x, prev_h], dim=-1)
        # 基础GRU计算
        z = torch.sigmoid(self.update_gate(combined))
        r = torch.sigmoid(self.reset_gate(combined))
        combined_reset = torch.cat([x, r * prev_h], dim=-1)
        h_tilde = torch.tanh(self.candidate(combined_reset))
        # 注意力权重计算
        attn_weight = torch.sigmoid(self.attention(h_tilde))
        # 用注意力调整更新门
        z = attn_weight * z
        h = (1 - z) * prev_h + z * h_tilde
        return h


# 初始化模型
lstm = LSTMModel(input_dim=3, hidden_dim=64)
gru = GRUModel(input_dim=3, hidden_dim=64)
augru = AUGRUModel(input_dim=3, hidden_dim=64)

# 输入数据 (时序长度=10, 特征=3)
x = torch.randn(10, 3)  # 10个时间步的3维特征
h_lstm, c_lstm = torch.zeros(64), torch.zeros(64)
h_gru = torch.zeros(64)
h_augru = torch.zeros(64)

# 逐步计算
for t in range(10):
    h_lstm, c_lstm = lstm(x[t], h_lstm, c_lstm)  # LSTM需维护c
    h_gru = gru(x[t], h_gru)                     # GRU只有h
    h_augru = augru(x[t], h_augru)               # AUGRU带注意力


print("----------------------------------")
print("LSTM h_lstm = ", h_lstm)
print("LSTM c_lstm = ", c_lstm)

print("----------------------------------")
print("h_gru = ", h_gru)

print("----------------------------------")
print("h_augru = ", h_augru)

#------------------------------------------------------
#another augerimplementation
#------------------------------------------------------
class AUGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wrz = nn.Linear(input_size, 2 * hidden_size)
        self.Urz = nn.Linear(hidden_size, 2 * hidden_size)
        self.Wh = nn.Linear(input_size, hidden_size)
        self.Uh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h_prev, att_score):
        rz = torch.sigmoid(self.Wrz(x) + self.Urz(h_prev))  # [batch, 2*hidden]
        r, z = torch.chunk(rz, 2, dim=-1)

        h_tilde = torch.tanh(self.Wh(x) + self.Uh(r * h_prev))

        # Apply attention score to update gate z
        z = z * att_score.unsqueeze(-1)  # [batch, hidden]

        h = (1 - z) * h_prev + z * h_tilde
        return h


batch_size = 4
seq_len = 6
input_size = 8
hidden_size = 16

x_seq = torch.randn(batch_size, seq_len, input_size)
att_scores = torch.rand(batch_size, seq_len)  # 注意力分数 ∈ [0, 1]

augru_cell = AUGRUCell(input_size, hidden_size)
h = torch.zeros(batch_size, hidden_size)

outputs = []
for t in range(seq_len):
    h = augru_cell(x_seq[:, t], h, att_scores[:, t])
    outputs.append(h.unsqueeze(1))

output_seq = torch.cat(outputs, dim=1)  # [batch, seq_len, hidden]
print("----------------------------------")
print("output_seq = ", output_seq)
