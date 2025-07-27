##RAG（Retrieval-Augmented Generation）在实际应用中可能因检索内容或生成逻辑而引入偏见或错误信息。为了减少偏见和错误信息，可以从以下几个方面进行改进和防护：
##
##✅ 1. 提高检索质量：减少检索阶段引入的偏见
##方法：
##高质量数据源：确保使用的知识库、文档来源权威、无偏。
##✅ 例如：政府网站、学术出版物、知名机构文档。
##多样化数据源：避免单一信息源带来的倾向性。
##✅ 例如：多种语言、多地区、多角度的内容。
##过滤和预处理：
##清除广告、虚假、过时或极端内容。
##去重、统一格式、移除垃圾文本。
##文档评分机制：在向量库中添加「可信度」、「更新时间」等辅助分数，优先检索高质量文档。
##
##✅ 2. 改进生成阶段（Generation）：增强判断力和鲁棒性
##方法：
##添加反事实约束 / 校验提示词：
##使用 prompt 工程让模型自我反思，如：
##“请只基于引用文档回答问题，如果无法确定，请回答‘无法确定’。”
##
##使用 Chain-of-Thought 或工具反推检索内容：
##提示模型验证信息：“文档A中的哪一句支持这个结论？”
##
##结合结构化数据或符号推理：
##将结构化数据（如数据库或知识图谱）与 LLM 输出结合，提高准确率。
##
##✅ 3. 输出可信度和引用信息
##方法：
##强制引用文档：
##格式化输出如：“根据文档[3]，XXX”
##
##计算并输出置信度分数：
##基于向量匹配得分、模型判断等，输出每条回答的置信度。
##
##让用户自己判断：显示相关原文段落：
##显示生成内容所参考的原文，有助于人工评估真假。
##
##✅ 4. 使用人类反馈或自动化评估机制
##方法：
##人工评估与反馈机制（HF/RA）：
##采集用户对回答的准确性反馈，用于微调或过滤错误模式。
##
##自动化评估器（Fact-checking model）：
##在最终输出前，用一个判别模型判断生成内容是否「被支持」「被反驳」「未提及」。
##
##✅ 5. 训练时加入对抗训练或公平性控制
##方法：
##对抗训练（Adversarial fine-tuning）：
##提供一些典型偏见样本，让模型学会拒绝或识别偏见。
##
##领域适配（Domain-specific RAG）：
##针对法律、医疗等高风险领域训练更稳健的RAG系统。
##
##引入 fairness-aware loss function：
##在微调时使用惩罚偏见生成的损失项。
##
##✅ 小结：五大策略概览
##类别	方法
##📚 检索层	数据去偏、文档可信度筛选、多源融合
##🧠 生成层	Prompt 校验、自我反思、引用限制
##📎 输出层	引用原文、置信度评分
##🧪 评估层	人工反馈、事实检查器
##🧬 训练层	对抗样本、行业适配、去偏损失函数


# install: pip install faiss-cpu sentence-transformers transformers

from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
#import torch
#torch.set_default_device("cpu")

# Step 1: 准备语料库
docs = [
    "The Eiffel Tower is located in Paris.",
    "The Great Wall of China is visible from space. (⚠️ FALSE)",
    "Python is a popular programming language.",
]

# Step 2: 构建向量索引
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(docs, convert_to_numpy=True)

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# Step 3: 查询输入
query = "Is the Great Wall of China visible from space?"
query_embedding = embedder.encode([query])

# Step 4: 相似度检索
D, I = index.search(np.array(query_embedding), k=2)
retrieved_docs = [docs[i] for i in I[0]]

# Step 5: 构建 Prompt + 调用 LLM
context = "\n".join(retrieved_docs)
question = f"""Answer the question based only on the facts below. If unsure, say "I don't know."

Context:
{context}

Question: {query}
Answer:"""

#qa = pipeline("text-generation", model="gpt2")
qa = pipeline("text-generation", model="gpt2", device=-1)
result = qa(question, max_new_tokens=50, do_sample=False)[0]['generated_text']

print(result)
