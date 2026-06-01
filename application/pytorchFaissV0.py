import numpy as np
import torch
import faiss  # pip install faiss-cpu or faiss-gpu

# Step 1: 构造向量库（10000 个向量，每个向量 128 维）
num_vectors = 10000
embedding_dim = 128

torch.manual_seed(42)
database_vectors = torch.randn(num_vectors, embedding_dim)  # (10000, 128)

# Step 2: 构建 Faiss 索引（使用 L2 距离或 Inner Product）
index = faiss.IndexFlatL2(embedding_dim)  # 你也可以用 IndexFlatIP（内积）
index.add(database_vectors.numpy())  # 加入向量库

print(f"向量库大小: {index.ntotal}")

# Step 3: 给定一个查询向量，查找最相似的 top-k 向量
query_vector = torch.randn(1, embedding_dim)  # 随机查询向量
k = 5  # 查前5个最相似向量

# 执行搜索
distances, indices = index.search(query_vector.numpy(), k)

# Step 4: 输出结果
print("查询向量最近的向量索引：", indices[0])
print("对应距离：", distances[0])

# 可视化比对
print("\n查询向量：")
print(query_vector.numpy()[0][:5], "...")

print("\n最近的第一个向量（部分展示）：")
print(database_vectors[indices[0][0]][:5], "...")
