import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
from tqdm import tqdm

# ====== 设置模型 ======
teacher_name = "bert-base-uncased"
student_name = "distilbert-base-uncased"
num_labels = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_name, num_labels=num_labels).to(device)
teacher_model.eval()

student_model = AutoModelForSequenceClassification.from_pretrained(student_name, num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(student_name)

# ====== PEFT 配置 ======
#peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"]  # DistilBERT 的注意力中常见模块
)

student_model = get_peft_model(student_model, peft_config).to(device)

for name, module in student_model.named_modules():
    if "lin" in name or "attention" in name:
        print(name)

# ====== 数据预处理 ======
dataset = load_dataset("imdb")
def preprocess(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)

encoded_dataset = dataset.map(preprocess, batched=True)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
train_dataloader = DataLoader(encoded_dataset["train"].shuffle(seed=42).select(range(2000)), batch_size=16)
test_dataloader = DataLoader(encoded_dataset["test"].select(range(500)), batch_size=32)

# ====== 蒸馏函数 ======
def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=2.0):
    loss_ce = F.cross_entropy(student_logits, labels)
    loss_kd = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean"
    ) * (temperature ** 2)
    return alpha * loss_ce + (1 - alpha) * loss_kd

# ====== 优化器 ======
optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
num_training_steps = len(train_dataloader) * 3
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# ====== 训练循环 ======
student_model.train()
for epoch in range(3):
    total_loss = 0
    loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            teacher_logits = teacher_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits

        student_outputs = student_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = distillation_loss(student_outputs.logits, teacher_logits, batch["label"])

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} average loss: {total_loss / len(train_dataloader):.4f}")

# ====== 测试评估 ======
student_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if "label" in batch:
            batch["labels"] = batch.pop("label")
        outputs = student_model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

print(f"\n🎯 Accuracy on test set: {correct / total:.2%}")
