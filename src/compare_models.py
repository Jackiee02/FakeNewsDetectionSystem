import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer
from datasets import DatasetDict
from tqdm import tqdm

from data_loader import load_and_prepare
from lstm_model import LSTMClassifier
from bilstm_model import BiLSTMClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"
EPOCHS = 15

eval_dir = "outputs/Evaluation"
log_dir = "outputs/Results/log"
model_dir = "outputs/Results/model"
os.makedirs(eval_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# ✅ 24 experimental combinations (2 architectures × 3 batch sizes × 4 learning rates)
param_grid = []
batch_sizes = [16, 32, 64]
learning_rates = [1e-2, 1e-3, 1e-4, 5e-4]

for bs in batch_sizes:
    for lr in learning_rates:
        # Small model architecture
        param_grid.append({
            "embedding_dim": 100,
            "hidden_dim": 128,
            "dropout": 0.3,
            "batch_size": bs,
            "lr": lr
        })
        # Big model architecture
        param_grid.append({
            "embedding_dim": 200,
            "hidden_dim": 256,
            "dropout": 0.3,
            "batch_size": bs,
            "lr": lr
        })

model_types = {
    "LSTM": LSTMClassifier,
    "BiLSTM": BiLSTMClassifier
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset: DatasetDict = load_and_prepare()
dataset = dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=128), batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'label'])

def get_loaders(batch_size):
    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset['validation'], batch_size=batch_size)
    return train_loader, val_loader

def evaluate_model(model, val_loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch['input_ids'].to(DEVICE), batch['label'].to(DEVICE)
            out = model(x)
            preds += out.argmax(1).tolist()
            labels += y.tolist()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return acc, f1

def train_model(model, train_loader, val_loader, optimizer, criterion, label_prefix):
    loss_log, val_loss_log = [], []
    train_acc_log, train_f1_log = [], []
    val_acc_log, val_f1_log = [], []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        train_preds, train_labels = [], []

        for batch in tqdm(train_loader, desc=f"{label_prefix} | Epoch {epoch+1}"):
            x, y = batch['input_ids'].to(DEVICE), batch['label'].to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_preds += out.argmax(dim=1).tolist()
            train_labels += y.tolist()

        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)

        # ✅ validation loss and metrics
        model.eval()
        val_preds, val_labels = [], []
        val_loss_total = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch['input_ids'].to(DEVICE), batch['label'].to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                val_loss_total += loss.item()
                val_preds += out.argmax(dim=1).tolist()
                val_labels += y.tolist()

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)

        # ✅ save all logs
        loss_log.append(total_loss)
        val_loss_log.append(val_loss_total)
        train_acc_log.append(train_acc)
        train_f1_log.append(train_f1)
        val_acc_log.append(val_acc)
        val_f1_log.append(val_f1)

        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Val_Loss={val_loss_total:.4f}, Train_Acc={train_acc:.4f}, Val_Acc={val_acc:.4f}, Val_F1={val_f1:.4f}")

    # ✅ write to log file
    pd.DataFrame({
        "epoch": list(range(1, EPOCHS + 1)),
        "train_loss": loss_log,
        "val_loss": val_loss_log,
        "train_accuracy": train_acc_log,
        "train_f1_score": train_f1_log,
        "val_accuracy": val_acc_log,
        "val_f1_score": val_f1_log
    }).to_csv(os.path.join(log_dir, f"{label_prefix}.csv"), index=False)

    return val_f1_log[-1]


def compare_all():
    results = []
    best_f1 = 0
    best_model = None
    best_label = ""

    for model_name, ModelClass in model_types.items():
        for params in param_grid:
            emb = params["embedding_dim"]
            hid = params["hidden_dim"]
            dropout = params["dropout"]
            batch_size = params["batch_size"]
            lr = params["lr"]

            label = f"{model_name}_emb{emb}_hid{hid}_bs{batch_size}_lr{lr}_do{dropout}"
            print(f"\n🚀 Training: {label}")

            train_loader, val_loader = get_loaders(batch_size)
            model = ModelClass(
                vocab_size=tokenizer.vocab_size,
                embedding_dim=emb,
                hidden_dim=hid,
                output_dim=2,
                pad_idx=tokenizer.pad_token_id
            )
            if hasattr(model, 'dropout'):
                model.dropout.p = dropout

            model.to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            final_f1 = train_model(model, train_loader, val_loader, optimizer, criterion, label)

            results.append({
                "Model": model_name,
                "Embedding Dim": emb,
                "Hidden Dim": hid,
                "Dropout": dropout,
                "Batch Size": batch_size,
                "Learning Rate": lr,
                "Validation F1": final_f1,
                "Label": label
            })

            torch.save(model.state_dict(), os.path.join(model_dir, f"{label}.pth"))

            if final_f1 > best_f1:
                best_f1 = final_f1
                best_model = model
                best_label = label

    if best_model:
        torch.save(best_model.state_dict(), os.path.join(model_dir, "best_model.pth"))
        print(f"\n✅ The best model has been saved as best_model.pth（{best_label}，Val F1: {best_f1:.4f}）")

    pd.DataFrame(results).to_csv(os.path.join(log_dir, "model_comparison_results.csv"), index=False)

if __name__ == "__main__":
    compare_all()
