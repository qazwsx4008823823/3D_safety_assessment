import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np


ROOT_DIR = "/your/model/path/"
DATA_DIR = "/dataset/outputs/path/"
TRAIN_LIST = os.path.join(DATA_DIR, "train_list.txt")
TEST_LIST = os.path.join(DATA_DIR, "test_list.txt")



BATCH_SIZE = 32
EPOCHS = 7
PATIENCE = 5
LR = 1e-3
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbeddingDataset(Dataset):
    def __init__(self, sample_list):
        self.samples = []
        self.labels = []
        for name in sample_list:
            embed_path = os.path.join(DATA_DIR, name, "concat_embed.npy")
            if os.path.exists(embed_path):
                embedding = np.load(embed_path).astype(np.float32)
                label = int(name.split("_")[0])
                self.samples.append((embedding, label))
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label = self.samples[idx]
        return torch.tensor(data), torch.tensor(label, dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# ========= 残差 MLP =========
class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.2)
        self.final = nn.Linear(512, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        residual = x  
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = x + residual  
        x = self.dropout(x)
        return self.final(x).squeeze(1)



def load_names(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]



def get_weighted_sampler(dataset):
    labels = torch.tensor(dataset.labels)
    class_counts = torch.bincount(labels.long())
    weights = len(dataset) / (2.0 * class_counts)
    sample_weights = weights[labels.long()]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)


def train():
    train_list = load_names(TRAIN_LIST)
    test_list = load_names(TEST_LIST)

    train_dataset = EmbeddingDataset(train_list)
    test_dataset = EmbeddingDataset(test_list)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=get_weighted_sampler(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = ResidualMLP().to(DEVICE)
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_loss = float('inf')
    patience = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                pred = (torch.sigmoid(out) > 0.4).float() 
                correct += (pred == y).sum().item()
                total += len(y)

        acc = correct / total if total > 0 else 0
        print(f"[Epoch {epoch}] Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience = PATIENCE
            save_path = os.path.join(ROOT_DIR, "model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to: {save_path}")
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping triggered")
                break

if __name__ == "__main__":
    train()
