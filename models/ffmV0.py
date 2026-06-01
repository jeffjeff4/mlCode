import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. 模拟数据和参数 ---
FEATURE_VOCAB_SIZES = {
    'user_id': 1000,
    'ad_id': 500,
    'item_category': 100,
    'advertiser_id': 50
}

FEATURE_FIELDS = {
    'user_id': 'user',
    'ad_id': 'ad',
    'item_category': 'item',
    'advertiser_id': 'advertiser'
}

ALL_FIELDS = list(set(FEATURE_FIELDS.values()))
NUM_FIELDS = len(ALL_FIELDS)

EMBEDDING_DIM = 10

BATCH_SIZE = 4
dummy_input_data = {
    'user_id': torch.randint(0, FEATURE_VOCAB_SIZES['user_id'], (BATCH_SIZE,)),
    'ad_id': torch.randint(0, FEATURE_VOCAB_SIZES['ad_id'], (BATCH_SIZE,)),
    'item_category': torch.randint(0_0, FEATURE_VOCAB_SIZES['item_category'], (BATCH_SIZE,)),
    'advertiser_id': torch.randint(0, FEATURE_VOCAB_SIZES['advertiser_id'], (BATCH_SIZE,)),
}
dummy_labels = torch.randint(0, 2, (BATCH_SIZE,)).float()

LEARNING_RATE = 0.01
NUM_EPOCHS = 10


class FFM(nn.Module):
    def __init__(self, feature_vocab_sizes, feature_fields, embedding_dim):
        super(FFM, self).__init__()

        self.feature_vocab_sizes = feature_vocab_sizes
        self.feature_fields = feature_fields
        self.all_fields = list(set(feature_fields.values()))
        self.num_fields = len(self.all_fields)
        self.embedding_dim = embedding_dim

        self.field_to_idx = {field: i for i, field in enumerate(self.all_fields)}

        self.linear_embeddings = nn.ModuleDict()
        for feature_name, vocab_size in feature_vocab_sizes.items():
            self.linear_embeddings[feature_name] = nn.Embedding(vocab_size, 1)

        self.bias = nn.Parameter(torch.zeros(1))

        self.feature_field_embeddings = nn.ModuleDict()
        for feature_name, vocab_size in feature_vocab_sizes.items():
            self.feature_field_embeddings[feature_name] = nn.ModuleDict()
            for field_name in self.all_fields:
                self.feature_field_embeddings[feature_name][field_name] = \
                    nn.Embedding(vocab_size, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        for embedding_layer in self.linear_embeddings.values():
            nn.init.xavier_uniform_(embedding_layer.weight)

        for feature_dict in self.feature_field_embeddings.values():
            for embedding_layer in feature_dict.values():
                nn.init.xavier_uniform_(embedding_layer.weight)

    def forward(self, input_data):
        batch_size = next(iter(input_data.values())).size(0)

        # 1. 计算线性项 (w_0 + sum(w_i * x_i))
        # CORRECTED: Initialize linear_term by expanding bias, then use non-in-place addition
        linear_term = self.bias.expand(batch_size, 1)  # Expand bias to match batch_size

        for feature_name, feature_indices in input_data.items():
            linear_term = linear_term + self.linear_embeddings[feature_name](feature_indices)  # Use non-in-place '+'

        # 2. 计算交叉项 (sum(sum(<v_{i,Fj}, v_{j,Fi}> * x_i * x_j)))
        cross_term = torch.zeros(batch_size, 1, device=linear_term.device)

        feature_names = list(input_data.keys())

        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                feature_name_i = feature_names[i]
                feature_name_j = feature_names[j]

                indices_i = input_data[feature_name_i]
                indices_j = input_data[feature_name_j]

                field_i = self.feature_fields[feature_name_i]
                field_j = self.feature_fields[feature_name_j]

                v_i_Fj = self.feature_field_embeddings[feature_name_i][field_j](indices_i)
                v_j_Fi = self.feature_field_embeddings[feature_name_j][field_i](indices_j)

                dot_product = torch.sum(v_i_Fj * v_j_Fi, dim=1, keepdim=True)

                cross_term += dot_product  # This `+=` is fine because cross_term is NOT a leaf variable

        logits = linear_term + cross_term

        return logits.squeeze(1)


# --- 3. 训练模型 ---
model = FFM(FEATURE_VOCAB_SIZES, FEATURE_FIELDS, EMBEDDING_DIM)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print(f"Starting FFM training with EMBEDDING_DIM={EMBEDDING_DIM}")

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()

    predictions_logits = model(dummy_input_data)

    loss = criterion(predictions_logits, dummy_labels)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")

print("\nTraining complete.")

# --- 4. 预测示例 ---
print("\n--- Prediction Example ---")
new_sample_input = {
    'user_id': torch.tensor([123]),
    'ad_id': torch.tensor([56]),
    'item_category': torch.tensor([12]),
    'advertiser_id': torch.tensor([5]),
}

model.eval()
with torch.no_grad():
    new_sample_logits = model(new_sample_input)
    predicted_prob = torch.sigmoid(new_sample_logits).item()

print(f"Predicted CTR for new sample: {predicted_prob:.4f}")