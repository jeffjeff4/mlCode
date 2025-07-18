import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Simulated YouTube video metadata
videos = pd.DataFrame({
    'video_id': [1, 2, 3],
    'title': ["How to Cook Perfect Pasta", "Italian Pasta Recipe", "Tech Review"],
    'description': ["Step-by-step pasta tutorial for beginners", "Learn authentic Italian pasta cooking",
                    "Latest tech gadgets"],
    'tags': ["pasta, cooking, tutorial", "Italian, pasta, recipe", "tech, review"],
    'url': ["youtube.com/123", "youtube.com/456", "youtube.com/789"]
})


# Step 1: Pre-process and index video metadata
def preprocess_videos(videos):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Combine title, description, tags
    texts = videos['title'] + ". " + videos['description'] + ". " + videos['tags']
    embeddings = model.encode(texts.tolist(), convert_to_numpy=True)

    # Create FAISS index
    d = embeddings.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return model, index, videos


# Step 2: Retrieve top-K videos
def retrieve_videos(query, model, index, videos, k=5):
    query_emb = model.encode([query])[0]
    distances, indices = index.search(np.array([query_emb]), k)
    return videos.iloc[indices[0]]


# Step 3: Generate response (simulated LLM)
def generate_response(query, retrieved_videos):
    prompt = f"User Query: {query}\nRetrieved Videos:\n"
    for idx, row in retrieved_videos.iterrows():
        prompt += f"{idx + 1}. Title: {row['title']}, Description: {row['description']}, URL: {row['url']}\n"
    prompt += "Task: Recommend the best video for the query and explain why."

    # Simulated LLM response (replace with actual LLM like Grok)
    response = (
        f"For the query '{query}', I recommend '{retrieved_videos.iloc[0]['title']}' "
        f"({retrieved_videos.iloc[0]['url']}). It provides a clear, beginner-friendly guide "
        f"that matches your request. Alternatively, '{retrieved_videos.iloc[1]['title']}' "
        f"({retrieved_videos.iloc[1]['url']}) offers additional insights."
    )
    return response


# Run RAG pipeline
if __name__ == "__main__":
    # Pre-process videos
    model, index, videos = preprocess_videos(videos)

    # User query
    query = "cooking pasta tutorials"

    # Retrieve
    retrieved_videos = retrieve_videos(query, model, index, videos, k=2)

    # Generate
    response = generate_response(query, retrieved_videos)
    print(response)