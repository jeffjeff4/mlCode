import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
#from datasets import load_dataset, load_metric

from datasets import load_dataset
import evaluate

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. 加载数据集（情感分类任务）
dataset = load_dataset("imdb")
dataset = dataset.shuffle(seed=42)
small_train_dataset = dataset["train"].select(range(2000))
small_test_dataset = dataset["test"].select(range(1000))

# 2. 加载 tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=256)

tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

# 3. 加载模型（DistilBERT + 分类头）
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4. 定义训练参数
##training_args = TrainingArguments(
##    output_dir="./sft_output",
##    evaluation_strategy="epoch",
##    save_strategy="epoch",
##    num_train_epochs=3,
##    per_device_train_batch_size=16,
##    per_device_eval_batch_size=16,
##    learning_rate=5e-5,
##    weight_decay=0.01,
##    logging_dir="./logs",
##    logging_steps=10,
##)


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
)

# 5. 定义评估函数
#metric = load_metric("accuracy")
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return metric.compute(predictions=predictions, references=labels)

# 6. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

model = model.to(device)

# 7. 训练
trainer.train()

# 8. 推理示例
def infer(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    label = "Positive" if prediction == 1 else "Negative"
    return label

print(infer("This movie was amazing!"))
print(infer("I don't like this film at all."))
