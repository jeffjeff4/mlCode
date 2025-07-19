import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1. 创建 toy 数据
data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'green', 'blue', 'red', 'green', 'blue', 'red'],
    'feature': [1.2, 0.3, 0.4, 1.1, 0.5, 0.2, 1.3, 0.6, 0.1, 1.0],
    'label':   [1,   0,    0,    1,    1,   0,   1,    0,   0,   1]
})

# 2. 将分类特征转为 categorical 类型
data['color'] = data['color'].astype('category')

# 3. 特征/标签分离
X = data[['color', 'feature']]
y = data['label']

# 4. 转换为 DMatrix
dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)

# 5. 训练参数
params = {
    'max_depth': 3,
    'eta': 1,
    'objective': 'binary:logistic',
    'tree_method': 'hist',              # 必须为 hist
    'eval_metric': 'auc',
    'verbosity': 1,
}

# 6. 训练模型
bst = xgb.train(params, dtrain, num_boost_round=5)

# 7. 打印模型结构（观察 split）
print("Model structure:")
print(bst.get_dump()[0])
