import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from collections import deque
import random

# Simulated YouTube video metadata
videos = pd.DataFrame({
    'video_id': [101, 202, 301, 302],
    'title': ["Pasta Tutorial", "Italian Pasta Recipe", "Italian Cooking Masterclass", "Pasta Carbonara Tutorial"],
    'description': ["Step-by-step pasta tutorial", "Authentic Italian pasta", "Learn Italian cooking",
                    "Carbonara recipe"],
    'tags': ["pasta, cooking, tutorial", "Italian, pasta, recipe", "Italian, cooking", "pasta, recipe"],
    'url': ["youtube.com/101", "youtube.com/202", "youtube.com/301", "youtube.com/302"]
})


# DQN for decision-making
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# Agent
class YouTubeAgent:
    def __init__(self, num_videos, embed_dim=384, memory_size=10000, batch_size=32, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        self.num_videos = num_videos
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Models
        self.model = DQN(embed_dim, num_videos).to(self.device)
        self.target_model = DQN(embed_dim, num_videos).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # Video embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.video_index, self.video_embs = self._index_videos(videos)

    def _index_videos(self, videos):
        texts = videos['title'] + ". " + videos['description'] + ". " + videos['tags']
        embeddings = self.embedding_model.encode(texts.tolist(), convert_to_numpy=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, embeddings

    def perceive(self, user_history, query=None):
        # User state: average of watched video embeddings + query embedding
        if user_history:
            history_embs = self.embedding_model.encode(
                [videos[videos['video_id'] == vid]['title'].iloc[0] for vid in user_history])
            user_state = np.mean(history_embs, axis=0)
        else:
            user_state = np.zeros(self.embed_dim)
        if query:
            query_emb = self.embedding_model.encode([query])[0]
            user_state = (user_state + query_emb) / 2
        return torch.tensor(user_state, dtype=torch.float32, device=self.device)

    def act(self, state, k=10):
        # Retrieve candidates
        distances, indices = self.video_index.search(np.array([state.cpu().numpy()]), k=100)
        candidates = indices[0]

        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.sample(list(candidates), k)
        else:
            state = state.unsqueeze(0)
            q_values = self.model(state)[:, candidates]
            _, top_k = torch.topk(q_values, k=k, dim=1)
            return candidates[top_k.cpu().numpy()[0]]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())


# Simulate agent interaction
if __name__ == "__main__":
    agent = YouTubeAgent(num_videos=len(videos))
    user_history = [101, 202]  # Watched videos
    query = "Italian recipes"

    # Perceive
    state = agent.perceive(user_history, query)

    # Act
    recommended_videos = agent.act(state, k=2)
    print("Recommended Videos:")
    for vid in recommended_videos:
        video = videos[videos['video_id'] == vid]
        print(f"- {video['title'].iloc[0]} ({video['url'].iloc[0]})")

    # Simulate feedback (e.g., user watches video 301 for 4 minutes)
    reward = 4.0  # Watch time in minutes
    next_state = agent.perceive(user_history + [301], query)
    agent.remember(state, recommended_videos[0], reward, next_state, False)
    agent.learn()
    agent.update_target()