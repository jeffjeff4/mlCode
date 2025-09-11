import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import faiss
import numpy as np

# 假设我们有以下文档库
documents = [
    "The Eiffel Tower is in Paris.",
    "The Great Wall of China is visible from space.",
    "Albert Einstein developed the theory of relativity.",
    "The moon is made of cheese.",  # 故意的错误
]

# Step 1: 使用简单词向量构造文档向量（实际应使用 BERT、MiniLM 等）
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.transformer(**inputs).last_hidden_state
        # 简单做平均池化
        return outputs.mean(dim=1).squeeze().numpy()

doc_embeddings = np.vstack([get_embedding(doc) for doc in documents]).astype("float32")

# Step 2: 构建 FAISS 索引
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Step 3: 提问并检索
query = "Who created relativity?"
query_vector = get_embedding(query).reshape(1, -1)
D, I = index.search(query_vector, k=2)

# Step 4: 构造 prompt（将相关文档拼接）
retrieved_docs = [documents[i] for i in I[0]]
context = "\n".join(retrieved_docs)
prompt = f"Context:\n{context}\n\nQ: {query}\nA:"

# Step 5: 生成回答
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=100, do_sample=True, top_p=0.95, top_k=50)
answer = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

print("Prompt:\n", prompt)
print("\nGenerated Answer:\n", answer)

# Step 6: 简单验证机制（避免 hallucination 或 bias）：
# 检查是否有回答中提及 context 未包含的内容
def is_supported(answer, context_docs):
    for sent in answer.split("."):
        if sent.strip() and not any(sent.strip() in doc for doc in context_docs):
            return False
    return True

print("\nIs the answer supported by context?", is_supported(answer, retrieved_docs))
