##python, generate a df,
##1) with 100 samples,
##2) each sample has 6 columns, they are user_id, item_id, ##is_viewed, is_clicked, is_bought, price, short setence for ##review, grade, time_stamp.
##3）user_id, item_id are randome id, could be duplicated
##4) is_viewed, is_clicked, is_bought are randomly generated, ##value either be 0 or 1
##5) price is the money for the product
##6) short setence for review, is a short sentence to tell ##whether the product is good or bad
##7) grade from an int of which the value is from 0 to 5
##8) generate a new column, named 'label'. its value = p0 * df['is_viewed'] + p1 * df['is_clicked'] + (1-p0-p1) * df[' is_bought']
##9) product group by user_id, sort by time stamp
##10) produce sequence features, e.g., sliding windows, click ##sequence, user browsing history
##11) pair positive samples and negative samples
##12) using ndcg loss as the objective function
##13) using ndcg as evaluation metrics
##14) create pytorch code for wide and deep model,
##15) train wide and deep model
##16) evaluate wide and deep model


import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.metrics import ndcg_score


# 1. 数据生成函数
def generate_data(num_samples=1000, p0=0.3, p1=0.3):
    np.random.seed(42)
    random.seed(42)

    user_ids = [f"user_{random.randint(1, 10)}" for _ in range(num_samples)]
    item_ids = [f"item_{random.randint(1, 20)}" for _ in range(num_samples)]

    data = {
        'user_id': user_ids,
        'item_id': item_ids,
        'is_viewed': np.random.randint(0, 2, num_samples),
        'is_clicked': np.random.randint(0, 2, num_samples),
        'is_bought': np.random.randint(0, 2, num_samples),
        'price': np.round(np.random.uniform(10, 500, num_samples), 2),
        'grade': np.random.randint(0, 6, num_samples),
        'time_stamp': [datetime.now() - timedelta(days=random.randint(0, 30))
                       for _ in range(num_samples)]
    }

    df = pd.DataFrame(data)
    df['label'] = p0 * df['is_viewed'] + p1 * df['is_clicked'] + (1 - p0 - p1) * df['is_bought']
    return df


# 2. 数据准备函数
def prepare_ranking_data(df):
    # 确保时间戳是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df['time_stamp']):
        df['time_stamp'] = pd.to_datetime(df['time_stamp'])

    # 按用户分组并按时间排序
    df = df.groupby('user_id').apply(lambda x: x.sort_values('time_stamp')).reset_index(drop=True)

    # 创建序列特征
    for col in ['is_viewed', 'is_clicked', 'is_bought', 'price', 'grade']:
        df[f'prev_{col}_mean'] = (
            df.groupby('user_id')[col]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
            .astype(np.float32)
        )

    # 标记正样本
    df['is_positive'] = df.groupby('user_id')['label'].transform(
        lambda x: (x >= x.quantile(0.7)).astype(int)
    )

    return df.fillna(0)


# 3. 自定义NDCG损失


# 4. 数据集类
class RankingDataset(Dataset):
    def __init__(self, df, user_encoder, item_encoder):
        self.df = df
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self._validate_data()

    def _validate_data(self):
        required = ['is_viewed', 'is_clicked', 'is_bought', 'price', 'grade',
                    'prev_is_viewed_mean', 'prev_is_clicked_mean',
                    'prev_is_bought_mean', 'prev_price_mean', 'prev_grade_mean']

        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing column: {col}")
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'wide': torch.FloatTensor(row[['is_viewed', 'is_clicked', 'is_bought']].values.astype(np.float32)),
            'deep': torch.FloatTensor(row[['price', 'grade',
                                           'prev_is_viewed_mean', 'prev_is_clicked_mean',
                                           'prev_is_bought_mean', 'prev_price_mean',
                                           'prev_grade_mean']].values.astype(np.float32)),
            'user': torch.LongTensor([self.user_encoder.transform([row['user_id']])[0]]),
            'item': torch.LongTensor([self.item_encoder.transform([row['item_id']])[0]]),
            'label': torch.FloatTensor([row['label']])
        }


# 5. Wide & Deep 模型
class WideAndDeep(nn.Module):
    def __init__(self, num_users, num_items):
        super().__init__()

        # Wide部分
        self.wide = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Deep部分
        self.user_embed = nn.Embedding(num_users, 16)
        self.item_embed = nn.Embedding(num_items, 16)

        self.deep = nn.Sequential(
            nn.Linear(7 + 32, 64),  # 7个特征 + 32维嵌入(16+16)
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )

        # 初始化所有参数
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def forward(self, wide, deep, user, item):
        # Wide部分
        wide_out = self.wide(wide)

        # Deep部分
        user_emb = self.user_embed(user).squeeze(1)
        item_emb = self.item_embed(item).squeeze(1)
        deep_in = torch.cat([deep, user_emb, item_emb], dim=1)
        deep_out = self.deep(deep_in)

        # 组合输出
        return torch.sigmoid(wide_out + deep_out)

# 1. 修改NDCG损失函数为可微分版本
class DifferentiableNDCGLoss(nn.Module):
    def __init__(self, k=5, temperature=0.1, device='cpu'):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.eps = 1e-8
        self.device = device

    def forward(self, y_pred, y_true):
        # 转换为张量并移动到设备
        y_pred = torch.as_tensor(y_pred, dtype=torch.float32, device=self.device).requires_grad_(True)
        y_true = torch.as_tensor(y_true, dtype=torch.float32, device=self.device)

        # 限制处理长度不超过k
        length = min(len(y_pred), self.k)

        # 使用softmax计算排序权重
        weights = torch.softmax(y_pred[:length] / self.temperature, dim=0)

        # 计算折扣因子
        discounts = 1.0 / torch.log2(torch.arange(2, length + 2, device=self.device))

        # 计算DCG
        dcg = torch.sum(weights * y_true[:length] * discounts)

        # 计算理想DCG
        ideal_dcg = torch.sum(torch.sort(y_true[:length], descending=True)[0] * discounts)

        # 返回负NDCG
        return -dcg / (ideal_dcg + self.eps)

# 6. 训练和评估函数
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        # 准备数据并确保需要梯度
        wide = batch['wide'].to(device).requires_grad_(True)
        deep = batch['deep'].to(device).requires_grad_(True)
        user = batch['user'].to(device)
        item = batch['item'].to(device)
        labels = batch['label'].to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        preds = model(wide, deep, user, item).squeeze()

        # 确保预测值需要梯度
        if not preds.requires_grad:
            preds = preds.requires_grad_(True)

        # 计算损失
        loss = criterion(preds, labels)

        # 反向传播
        loss.backward()

        # 梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            preds.extend(model(**inputs).cpu().numpy())
            labels.extend(batch['label'].cpu().numpy())
    #return ndcg_score([labels], [preds], k=5)
    #differentiableNDCGLoss = DifferentiableNDCGLoss(k=5, temperature=0.1)
    #return differentiableNDCGLoss.forward(labels, preds)
    return criterion(labels, preds)


# 7. 主程序
def main():
    # 生成和准备数据
    df = generate_data(1000)
    df = prepare_ranking_data(df)

    # 创建编码器
    user_encoder = LabelEncoder().fit(df['user_id'])
    item_encoder = LabelEncoder().fit(df['item_id'])

    # 划分数据集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 创建数据集
    train_dataset = RankingDataset(train_df, user_encoder, item_encoder)
    test_dataset = RankingDataset(test_df, user_encoder, item_encoder)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WideAndDeep(
        num_users=len(user_encoder.classes_),
        num_items=len(item_encoder.classes_)
    ).to(device)

    # 训练配置
    criterion = DifferentiableNDCGLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        ndcg = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch + 1}: Loss={train_loss:.4f}, NDCG@5={ndcg:.4f}")


if __name__ == "__main__":
    main()