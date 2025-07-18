import torch
import torch.nn as nn


# --- 1. 定義嵌入層和數值特徵處理 ---
class RecommendationModelWithConcatenation(nn.Module):
    def __init__(self, num_users, num_movies, user_embedding_dim, movie_embedding_dim):
        super(RecommendationModelWithConcatenation, self).__init__()

        # 用戶 ID 的嵌入層
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)

        # 電影 ID 的嵌入層
        self.movie_embedding = nn.Embedding(num_movies, movie_embedding_dim)

        # 假設我們有 2 個數值特徵：用戶年齡、電影平均評分
        # 我們不需要為它們定義額外的層，它們將直接作為輸入維度
        num_numerical_features = 2

        # 拼接後的新嵌入向量的總維度
        # user_embedding_dim + movie_embedding_dim + num_numerical_features
        concatenated_embedding_dim = user_embedding_dim + movie_embedding_dim + num_numerical_features

        # 後續的預測網絡 (例如一個簡單的多層感知機 MLP)
        self.fc_layers = nn.Sequential(
            nn.Linear(concatenated_embedding_dim, 128),  # 第一層
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),  # 第二層
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),  # 輸出層，預測一個分數 (例如點擊機率)
            nn.Sigmoid()  # 如果是預測點擊機率，使用 Sigmoid 激活函數
        )

    def forward(self, user_ids, movie_ids, user_ages, movie_ratings):
        # 1. 獲取類別特徵的嵌入向量
        user_embed = self.user_embedding(user_ids)  # (batch_size, user_embedding_dim)
        movie_embed = self.movie_embedding(movie_ids)  # (batch_size, movie_embedding_dim)

        # 2. 準備數值特徵
        # 確保數值特徵的形狀與 embedding 批次維度匹配
        # unsqueeze(1) 將 (batch_size,) 變為 (batch_size, 1)
        user_ages_tensor = user_ages.float().unsqueeze(1)  # (batch_size, 1)
        movie_ratings_tensor = movie_ratings.float().unsqueeze(1)  # (batch_size, 1)

        # 3. 拼接所有特徵
        # 使用 torch.cat 沿著特徵維度 (dim=1) 進行拼接
        # 拼接順序可以自己定義
        combined_features = torch.cat(
            [user_embed, movie_embed, user_ages_tensor, movie_ratings_tensor],
            dim=1  # 在第二個維度 (特徵維度) 上拼接
        )  # (batch_size, user_embedding_dim + movie_embedding_dim + 2)

        # 4. 將拼接後的向量傳入預測網絡
        prediction = self.fc_layers(combined_features)  # (batch_size, 1)

        return prediction.squeeze(1)  # 返回 (batch_size,) 的預測結果


# --- 模擬數據和模型初始化 ---
num_users = 10000
num_movies = 5000
user_embed_dim = 32
movie_embed_dim = 64

model = RecommendationModelWithConcatenation(num_users, num_movies, user_embed_dim, movie_embed_dim)

# --- 模擬一個批次的輸入數據 ---
batch_size = 4
# 假設用戶 ID 是 0 到 num_users-1
sample_user_ids = torch.randint(0, num_users, (batch_size,))
# 假設電影 ID 是 0 到 num_movies-1
sample_movie_ids = torch.randint(0, num_movies, (batch_size,))
# 模擬年齡 (18-60歲)
sample_user_ages = torch.randint(18, 61, (batch_size,))
# 模擬電影評分 (1.0-5.0)
sample_movie_ratings = torch.rand(batch_size) * 4 + 1  # 生成 1.0 到 5.0 之間的浮點數

print(f"Sample User IDs: {sample_user_ids}")
print(f"Sample Movie IDs: {sample_movie_ids}")
print(f"Sample User Ages: {sample_user_ages}")
print(f"Sample Movie Ratings: {sample_movie_ratings}")

# --- 執行前向傳播 ---
predictions = model(sample_user_ids, sample_movie_ids, sample_user_ages, sample_movie_ratings)

print(f"\n模型輸出預測 (點擊機率): {predictions}")
print(f"輸出預測的形狀: {predictions.shape}")

# 驗證拼接後向量的維度
# user_embed_dim (32) + movie_embed_dim (64) + 2 (數值特徵) = 98
# 這應該是第一個全連接層的輸入維度
print(f"預期拼接後向量維度: {user_embed_dim + movie_embed_dim + 2}")
print(f"模型第一個線性層的輸入維度: {model.fc_layers[0].in_features}")  # 檢查第一個FC層的in_features