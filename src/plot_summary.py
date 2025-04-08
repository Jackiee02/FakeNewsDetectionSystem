import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, f1_score
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch

from data_loader import load_and_prepare
from lstm_model import LSTMClassifier
from bilstm_model import BiLSTMClassifier

log_dir = "outputs/Results/log"
eval_dir = "outputs/Evaluation"
model_dir = "outputs/Results/model"
os.makedirs(os.path.join(eval_dir, "heatmaps"), exist_ok=True)
os.makedirs(os.path.join(eval_dir, "radar"), exist_ok=True)

def plot_f1_heatmap(model_name):
    files = [f for f in os.listdir(log_dir) if f.endswith(".csv") and f.startswith(model_name)]

    records = []
    for file in files:
        parts = file.replace(".csv", "").split("_")
        bs = int([p for p in parts if p.startswith("bs")][0][2:])
        lr_str = [p for p in parts if p.startswith("lr")][0][2:]
        lr = float(lr_str.replace("e", "e-"))
        df = pd.read_csv(os.path.join(log_dir, file))
        f1 = df["val_f1_score"].iloc[-1]
        records.append((bs, lr, f1))

    df = pd.DataFrame(records, columns=["Batch Size", "Learning Rate", "F1 Score"])
    df = df.groupby(["Batch Size", "Learning Rate"], as_index=False).max()

    pivot = df.pivot(index="Batch Size", columns="Learning Rate", values="F1 Score")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title(f"F1 Score Heatmap ({model_name})")
    plt.savefig(os.path.join(eval_dir, "heatmaps", f"f1_heatmap_{model_name.lower()}.png"))
    plt.close()
    print(f"✅ F1 heatmap saved: heatmaps/f1_heatmap_{model_name.lower()}.png")

def compute_validation_metrics(model_name, emb, hid, bs, lr):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_and_prepare()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    dataset = dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=128), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'label'])
    val_loader = DataLoader(dataset["validation"], batch_size=32)

    if model_name == "LSTM":
        model = LSTMClassifier(tokenizer.vocab_size, emb, hid, 2, tokenizer.pad_token_id)
    else:
        model = BiLSTMClassifier(tokenizer.vocab_size, emb, hid, 2, tokenizer.pad_token_id)

    # 🔧 Construct full model filename (with dropout)
    lr_str = str(lr)  # Use original string without formatting

    model_filename = f"{model_name}_emb{emb}_hid{hid}_bs{bs}_lr{lr_str}_do0.3.pth"
    model_path = os.path.join(model_dir, model_filename)

    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["input_ids"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            preds = probs.argmax(1)
            all_preds += preds.cpu().tolist()
            all_labels += y.cpu().tolist()
            all_probs += probs[:, 1].cpu().tolist()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)

    return {
        "F1 Score": f1,
        "Accuracy": acc,
        "ROC AUC": roc_auc,
        "PR AUC": pr_auc
    }

def plot_radar_chart():
    result_df = pd.read_csv(os.path.join(log_dir, "model_comparison_results.csv"))
    best_lstm = result_df[result_df["Model"] == "LSTM"].sort_values("Validation F1", ascending=False).iloc[0]
    best_bilstm = result_df[result_df["Model"] == "BiLSTM"].sort_values("Validation F1", ascending=False).iloc[0]

    print("🔍 Computing validation metrics for LSTM...")
    metrics_lstm = compute_validation_metrics(
        "LSTM", int(best_lstm["Embedding Dim"]), int(best_lstm["Hidden Dim"]),
        int(best_lstm["Batch Size"]), float(best_lstm["Learning Rate"])
    )

    print("🔍 Computing validation metrics for BiLSTM...")
    metrics_bilstm = compute_validation_metrics(
        "BiLSTM", int(best_bilstm["Embedding Dim"]), int(best_bilstm["Hidden Dim"]),
        int(best_bilstm["Batch Size"]), float(best_bilstm["Learning Rate"])
    )

    categories = list(metrics_lstm.keys())
    values1 = list(metrics_lstm.values())
    values2 = list(metrics_bilstm.values())

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values1 += values1[:1]
    values2 += values2[:1]
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values1, label="LSTM", linestyle="--")
    ax.fill(angles, values1, alpha=0.25)
    ax.plot(angles, values2, label="BiLSTM", linestyle="-")
    ax.fill(angles, values2, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_title("Model Comparison Radar Chart")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "radar", "model_comparison_radar.png"))
    plt.close()
    print("✅ Model radar chart saved: radar/model_comparison_radar.png")

if __name__ == "__main__":
    plot_f1_heatmap("LSTM")
    plot_f1_heatmap("BiLSTM")
    plot_radar_chart()
