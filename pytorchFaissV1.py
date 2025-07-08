import numpy as np
import faiss  # pip install faiss-cpu

# 设置随机种子
np.random.seed(42)

# -------------------------------
# Step 1: 构造向量库和查询向量
# -------------------------------
num_vectors = 10000
vector_dim = 128
num_queries = 5

# 模拟 embedding
database_vectors = np.random.randn(num_vectors, vector_dim).astype(np.float32)
query_vectors = np.random.randn(num_queries, vector_dim).astype(np.float32)

# -------------------------------
# Step 2: 构建不同 ANN 索引
# -------------------------------

# IVF（聚类量化）
nlist = 100  # 聚类中心数
quantizer = faiss.IndexFlatL2(vector_dim)  # 基础量化器
index_ivf = faiss.IndexIVFFlat(quantizer, vector_dim, nlist, faiss.METRIC_L2)
index_ivf.train(database_vectors)
index_ivf.add(database_vectors)

# PQ（乘积量化）
m = 16  # 子空间数
index_pq = faiss.IndexPQ(vector_dim, m, 8)
index_pq.train(database_vectors)
index_pq.add(database_vectors)

# IVF + PQ（粗量化 + 残差 PQ）
index_ivfpq = faiss.IndexIVFPQ(quantizer, vector_dim, nlist, m, 8)
index_ivfpq.train(database_vectors)
index_ivfpq.add(database_vectors)

# -------------------------------
# Step 3: 查询函数
# -------------------------------
def run_search(index, name, queries, topk=5):
    print(f"\n🔍 {name} 检索结果")
    if hasattr(index, 'nprobe'):
        index.nprobe = 10  # 控制检索的聚类数量
    D, I = index.search(queries, topk)
    for i in range(len(queries)):
        print(f"Query {i+1} Top-{topk} indices: {I[i]}, distances: {np.round(D[i], 3)}")

# -------------------------------
# Step 4: 测试查询
# -------------------------------
run_search(index_ivf, "IVF 聚类量化", query_vectors)
run_search(index_pq, "PQ 乘积量化", query_vectors)
run_search(index_ivfpq, "IVF+PQ 粗量+残差量化", query_vectors)
