import torch
from torch import nn

torch.cuda.memory._record_memory_history(max_entries = 100000)
torch.cuda.memory._record_memory_history()

model = nn.Linear(10_000, 50_000, device='cuda')
for _ in range(3):
    inputs = torch.randn(5_000, 10_000, device='cuda')
    outputs = model(inputs)

torch.cuda.memory._dump_snapshot("//Users//shizhefu0//Desktop//tmp//profile.pkl")
torch.cuda.memory._record_memory_history(enabled=None)

