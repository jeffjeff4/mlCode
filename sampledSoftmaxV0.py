import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- 1. 模拟数据和参数 ---
VOCAB_SIZE = 10000  # 词汇表大小 (假设输出类别很多)
EMBEDDING_DIM = 128  # 模型的特征维度
NUM_NEGATIVE_SAMPLES = 10  # 每个正例采样的负样本数量
LEARNING_RATE = 0.01
NUM_EPOCHS = 5

# 模拟类别频率 (用于采样概率 P_sample)
# 实际中，这可以是词频、物品流行度等
# 频率越高，被采样的概率越大
class_frequencies = torch.randint(1, 1000, (VOCAB_SIZE,)).float()
# 为了采样，通常使用一个平滑的、幂次采样的概率，如 Word2Vec 的 0.75 次方
# 这里我们用一个简化的 unigram 概率作为采样概率
sampling_probabilities = class_frequencies / class_frequencies.sum()
# 将采样概率转换为对数形式，因为在 Softmax 公式中会用到 log P_sample
log_sampling_probabilities = torch.log(sampling_probabilities + 1e-10)  # 加小常数避免log(0)


# --- 2. 定义模型 ---
# 假设我们有一个简单的线性模型，将输入特征映射到每个类别的得分
class SimpleClassifier(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(SimpleClassifier, self).__init__()
        # 假设输入是一个固定维度的特征向量 (例如，来自一个encoder)
        # 这里为了演示，我们模拟一个简单的线性层作为输出层
        self.output_layer = nn.Linear(embedding_dim, vocab_size, bias=False)  # 不带偏置项，简化

        # 初始化权重 (可选)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, input_features):
        # input_features: (batch_size, embedding_dim)
        # scores: (batch_size, vocab_size)
        scores = self.output_layer(input_features)
        return scores


# --- 3. 采样 Softmax 损失函数实现 ---
class SampledSoftmaxLoss(nn.Module):
    def __init__(self, model, log_sampling_probabilities, num_negative_samples):
        super(SampledSoftmaxLoss, self).__init__()
        self.model = model
        self.log_sampling_probabilities = log_sampling_probabilities
        self.num_negative_samples = num_negative_samples

    def forward(self, input_features, true_labels):
        batch_size = input_features.size(0)

        # --- 获取正例得分 ---
        # scores_all: (batch_size, vocab_size)
        scores_all = self.model(input_features)
        # Gather scores for true labels
        # true_labels: (batch_size,)
        # positive_scores: (batch_size,)
        positive_scores = scores_all.gather(1, true_labels.unsqueeze(1)).squeeze(1)

        # --- 获取正例的 log_P_sample(y_true) ---
        # log_true_sample_prob: (batch_size,)
        log_true_sample_prob = self.log_sampling_probabilities[true_labels]

        # --- 采样负例 ---
        # 从采样分布中选择负样本
        # multinomial returns indices based on probabilities
        # Here we sample num_negative_samples *per* example in the batch
        # This creates a flat list of negative indices, which we then reshape
        negative_indices = torch.multinomial(
            torch.exp(self.log_sampling_probabilities),  # multinomial expects probabilities
            batch_size * self.num_negative_samples,
            replacement=True  # 允许重复采样
        ).view(batch_size, self.num_negative_samples)

        # 确保采样的负例不包含正例（尽管在实践中，由于词汇表大，概率很小）
        # 这里为了简化，我们不进行显式去重或替换操作
        # 在 PyTorch 的 C++ 内部实现中，通常会更复杂地处理避免采样到正例的情况

        # --- 获取负例得分 ---
        # negative_scores: (batch_size, num_negative_samples)
        negative_scores = scores_all.gather(1, negative_indices)

        # --- 获取负例的 log_P_sample(y_neg) ---
        # log_negative_sample_prob: (batch_size, num_negative_samples)
        log_negative_sample_prob = self.log_sampling_probabilities[negative_indices]

        # --- 计算修正后的 logit (score - log P_sample) ---
        # Adjusted score for true labels: (batch_size,)
        adjusted_positive_scores = positive_scores - log_true_sample_prob

        # Adjusted scores for negative labels: (batch_size, num_negative_samples)
        adjusted_negative_scores = negative_scores - log_negative_sample_prob

        # Combine positive and negative adjusted scores for Softmax
        # concatenated_scores: (batch_size, 1 + num_negative_samples)
        # The true label is at index 0 for each example
        concatenated_scores = torch.cat(
            (adjusted_positive_scores.unsqueeze(1), adjusted_negative_scores), dim=1
        )

        # Create target labels for the sampled Softmax (all 0s, as true label is at index 0)
        # target_labels_for_softmax: (batch_size,) filled with 0s
        target_labels_for_softmax = torch.zeros(batch_size, dtype=torch.long, device=input_features.device)

        # Calculate cross-entropy loss on the sampled scores
        # F.cross_entropy handles log_softmax internally
        loss = F.cross_entropy(concatenated_scores, target_labels_for_softmax)

        return loss


# --- 4. 模拟训练过程 ---

# 创建模型
model = SimpleClassifier(EMBEDDING_DIM, VOCAB_SIZE)
criterion = SampledSoftmaxLoss(model, log_sampling_probabilities, NUM_NEGATIVE_SAMPLES)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Starting training with VOCAB_SIZE={VOCAB_SIZE}, EMBEDDING_DIM={EMBEDDING_DIM}")

# 模拟训练数据 (输入特征和对应的真实标签)
# input_features: 例如，来自一个预训练的文本编码器
# true_labels: 真实的分类标签 (词汇表中的索引)
dummy_input_features = torch.randn(1000, EMBEDDING_DIM)  # 1000个样本，每个128维特征
dummy_true_labels = torch.randint(0, VOCAB_SIZE, (1000,))  # 1000个样本，每个对应一个真实标签

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()

    # 随机选择一个批次
    indices = torch.randperm(dummy_input_features.size(0))[:64]  # Batch size 64
    batch_features = dummy_input_features[indices]
    batch_labels = dummy_true_labels[indices]

    loss = criterion(batch_features, batch_labels)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")

print("\nTraining complete.")

# --- 5. 验证 (简单示例) ---
# 训练后，我们可以尝试预测一个样本
# 注意：在采样 Softmax 训练后，如果你需要得到所有类别的概率，
# 在推理阶段通常需要计算完整的 Softmax（除非有其他近似方法）。
# 这里的验证只是为了演示模型学习了如何预测。

# 假设一个测试样本
test_features = torch.randn(1, EMBEDDING_DIM)  # 单个测试样本
with torch.no_grad():
    all_scores = model(test_features)  # 获取所有类别的分数

# 为了演示，我们在这里计算完整的 Softmax 来看看最高分是哪个类别
# 实际应用中，如果类别仍太多，会用其他近似推断方法
probabilities = F.softmax(all_scores, dim=1)
predicted_class = torch.argmax(probabilities, dim=1).item()

print(f"\nTest sample predicted class (full softmax for inference): {predicted_class}")
print(f"Top 5 predicted probabilities:")
top5_probs, top5_indices = torch.topk(probabilities, 5)
for i in range(5):
    print(f"  Class {top5_indices[0][i].item()}: {top5_probs[0][i].item():.4f}")