import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- 1. 模拟数据和参数 ---
VOCAB_SIZE = 1000  # 词汇表大小
EMBEDDING_DIM = 128  # 词嵌入维度
NUM_NEGATIVE_SAMPLES = 5  # 每个正例对应的负样本数量
LEARNING_RATE = 0.01
NUM_EPOCHS = 10

# 模拟词汇频率（用于噪声分布 P_n）
# 真实场景中，这是根据词在语料库中的出现频率计算的
# 这里假设某些词更常见
word_counts = torch.randint(10, 1000, (VOCAB_SIZE,)).float()
# 为了 NCE，通常使用 P_n(w) = count(w) / total_words 或 (count(w)^0.75) / sum(count^0.75)
# 这里我们用一个简化的 unigram 分布
noise_distribution = word_counts / word_counts.sum()


# --- 2. 定义模型 (词嵌入层) ---
class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(WordEmbeddingModel, self).__init__()
        # Input Embeddings (for center word)
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Output Embeddings (for context word)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 初始化嵌入权重 (可选，但通常有助于训练)
        self.input_embeddings.weight.data.uniform_(-1.0, 1.0)
        self.output_embeddings.weight.data.uniform_(-1.0, 1.0)

    def forward(self, center_word_idx, target_word_idx):
        # 获取中心词和目标词的嵌入
        center_embed = self.input_embeddings(center_word_idx)
        target_embed = self.output_embeddings(target_word_idx)

        # 计算点积相似度
        score = torch.sum(center_embed * target_embed, dim=-1)
        return score


# --- 3. NCE Loss 函数实现 ---
# 这里我们手动实现 NCE 的逻辑，PyTorch 也有 nn.NLLLoss 或 F.log_softmax 配合，
# 但为了清晰展示 NCE 思想，手动实现更直观。
# 实际使用中，通常会利用 PyTorch 内置的 NCE 或 sampled_softmax 功能。

class NCELoss(nn.Module):
    def __init__(self, model, noise_distribution, num_negative_samples):
        super(NCELoss, self).__init__()
        self.model = model
        self.noise_distribution = noise_distribution
        self.num_negative_samples = num_negative_samples

    def forward(self, center_word_idx, positive_word_idx):
        batch_size = center_word_idx.size(0)

        # --- 正例部分 ---
        # 计算正例分数 s(W_true, C)
        pos_score = self.model(center_word_idx, positive_word_idx)

        # 获取正例在噪声分布中的对数概率 log P_n(W_true)
        # noise_distribution是一个一维张量，直接用positive_word_idx索引
        log_pos_noise_prob = torch.log(self.noise_distribution[positive_word_idx] + 1e-10)  # 加一个小常数避免log(0)

        # NCE 目标中的 argument: s(W,C) - log(k) - log(P_n(W))
        # NCE 的目标是训练一个二分类器，判断样本是真实样本还是噪声样本。
        # P(D=1|W,C) = sigmoid(log(P_true(W|C)) - log(k*P_noise(W)))
        # 其中 log(P_true(W|C)) 实际就是我们模型学习到的 score(W,C)
        # 这里我们直接用 score(W,C) 作为 log P_true(W|C) 的近似
        # arg = pos_score - torch.log(torch.tensor(self.num_negative_samples, dtype=torch.float)) - log_pos_noise_prob
        # 对于正例，我们希望它被分类为真实数据 (label=1)
        # Log likelihood for positive samples: log(sigmoid(arg))

        # 简化版：直接使用目标公式的 log P_d(W|C) - log(k P_n(W))
        # 这里的 pos_score 对应 log(P_true(W|C))
        # 注意：这里的 K 是 number of noise samples PER positive sample
        pos_logit = pos_score - (self.num_negative_samples * log_pos_noise_prob)
        # NCE 损失公式 for positive samples
        pos_loss = -F.logsigmoid(pos_logit).mean()

        # --- 负例部分 ---
        # 从噪声分布中采样负例
        # replace=True 允许重复采样，适用于大词汇表
        negative_word_indices = torch.multinomial(
            self.noise_distribution,
            batch_size * self.num_negative_samples,
            replacement=True
        ).view(batch_size, self.num_negative_samples)

        # 获取负例分数 s(W_noise, C)
        # center_word_idx 需要扩展以匹配 negative_word_indices 的形状
        center_word_idx_expanded = center_word_idx.unsqueeze(1).expand(-1, self.num_negative_samples)
        neg_score = self.model(center_word_idx_expanded, negative_word_indices)

        # 获取负例在噪声分布中的对数概率 log P_n(W_noise)
        log_neg_noise_prob = torch.log(self.noise_distribution[negative_word_indices] + 1e-10)

        # NCE 目标中的 argument for negative samples
        neg_logit = neg_score - (self.num_negative_samples * log_neg_noise_prob)
        # 对于负例，我们希望它被分类为噪声数据 (label=0)
        # Log likelihood for negative samples: log(1 - sigmoid(arg)) = log(sigmoid(-arg))
        neg_loss = -F.logsigmoid(-neg_logit).mean()

        # 总损失是正例和负例损失之和
        total_loss = pos_loss + neg_loss
        return total_loss


# --- 4. 模拟训练过程 ---

# 创建模型
model = WordEmbeddingModel(VOCAB_SIZE, EMBEDDING_DIM)
criterion = NCELoss(model, noise_distribution, NUM_NEGATIVE_SAMPLES)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Starting training with VOCAB_SIZE={VOCAB_SIZE}, EMBEDDING_DIM={EMBEDDING_DIM}")

# 模拟训练数据 (中心词, 上下文词对)
# 假设我们有一些中心词和它们对应的上下文词
# 真实数据会从语料库中提取
dummy_center_words = torch.randint(0, VOCAB_SIZE, (1000,))
dummy_positive_words = torch.randint(0, VOCAB_SIZE, (1000,))

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()

    # 随机选择一个批次
    indices = torch.randperm(dummy_center_words.size(0))[:64]  # Batch size 64
    batch_center = dummy_center_words[indices]
    batch_positive = dummy_positive_words[indices]

    loss = criterion(batch_center, batch_positive)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")

print("\nTraining complete.")

# --- 5. 验证 (简单示例) ---
# 训练后，我们可以查看词嵌入的相似度
# 获取两个随机词的嵌入
word_id_1 = torch.tensor([10])
word_id_2 = torch.tensor([20])
word_id_3 = torch.tensor([30])

embed1 = model.input_embeddings(word_id_1)
embed2 = model.input_embeddings(word_id_2)
embed3 = model.input_embeddings(word_id_3)

# 计算余弦相似度
# 理想情况下，如果词1和词2在训练中经常共现，它们的相似度会更高
similarity_1_2 = F.cosine_similarity(embed1, embed2).item()
similarity_1_3 = F.cosine_similarity(embed1, embed3).item()

print(f"\nCosine similarity between word {word_id_1.item()} and word {word_id_2.item()}: {similarity_1_2:.4f}")
print(f"Cosine similarity between word {word_id_1.item()} and word {word_id_3.item()}: {similarity_1_3:.4f}")

# 注意：由于这是简化模拟，且数据是随机生成的，所以相似度结果没有实际意义，
# 但它展示了计算流程。在真实语料库训练后，相似度会有实际含义。