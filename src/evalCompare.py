import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from data_loader import load_and_prepare
from utils import evaluate_and_plot

log_dir = "outputs/Results/log"
output_root = "outputs/Evaluation"
model_path = "outputs/Results/model/best_model.pth"

def extract_params(file):
    parts = file.replace(".csv", "").split("_")
    return {
        "model": parts[0],
        "emb": parts[1],
        "hid": parts[2],
        "bs": [p for p in parts if p.startswith("bs")][0][2:],
        "lr": [p for p in parts if p.startswith("lr")][0][2:]
    }

def plot_nested_by_fixed_param(fixed_param):
    changing_param = "bs" if fixed_param == "lr" else "lr"
    log_files = [f for f in os.listdir(log_dir) if f.endswith(".csv") and "model_comparison_results" not in f]

    metric_pairs = [
        ("train_loss", "val_loss", "Loss", "loss.png"),
        ("train_accuracy", "val_accuracy", "Accuracy", "accuracy.png"),
        ("train_f1_score", "val_f1_score", "F1 Score", "f1_score.png"),
    ]

    for model_type in ["LSTM", "BiLSTM"]:
        for size_key, size_folder in [("emb100", "small_model"), ("emb200", "large_model")]:
            model_logs = [
                (f, extract_params(f)) for f in log_files
                if model_type in f and size_key in f
            ]

            grouped = {}
            for file, params in model_logs:
                fixed_val = params[fixed_param]
                grouped.setdefault(fixed_val, []).append((file, params[changing_param]))

            cmap = get_cmap("tab10")

            for fixed_val, files in grouped.items():
                changing_vals = [val for _, val in files]
                color_map = {val: cmap(i % 10) for i, val in enumerate(sorted(set(changing_vals)))}

                for train_metric, val_metric, display_name, filename in metric_pairs:
                    plt.figure(figsize=(10, 5))
                    plotted_labels = set()

                    for file, changing_val in files:
                        df = pd.read_csv(os.path.join(log_dir, file))
                        label = f"{changing_param}={changing_val}"
                        if label in plotted_labels:
                            continue
                        if train_metric in df.columns:
                            plt.plot(df["epoch"], df[train_metric], linestyle="--", color=color_map[changing_val], label=f"{label} - Train")
                        if val_metric in df.columns:
                            plt.plot(df["epoch"], df[val_metric], linestyle="-", color=color_map[changing_val], label=f"{label} - Val")
                        plotted_labels.add(label)

                    if plotted_labels:
                        plt.xlabel("Epoch")
                        plt.ylabel(display_name)
                        plt.title(f"{model_type} - {size_folder} - {display_name} | Fixed {fixed_param}={fixed_val}")
                        plt.grid(True)
                        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="small")
                        plt.tight_layout()

                        subfolder = "BatchSize" if fixed_param == "lr" else "LearningRate"
                        subdir = f"{fixed_param}{fixed_val}"
                        save_dir = os.path.join(output_root, model_type, size_folder, subfolder, subdir)
                        os.makedirs(save_dir, exist_ok=True)
                        plt.savefig(os.path.join(save_dir, filename))
                        print(f"✅ Image has been saved: {os.path.join(save_dir, filename)}")
                        plt.close()

def evaluate_best_model_on_validation():
    from lstm_model import LSTMClassifier
    from bilstm_model import BiLSTMClassifier
    from sklearn.metrics import accuracy_score, f1_score

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_and_prepare()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    dataset = dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=128), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'label'])
    val_loader = DataLoader(dataset['validation'], batch_size=32)

    result_df = pd.read_csv(os.path.join(log_dir, "model_comparison_results.csv"))
    best_row = result_df.sort_values("Validation F1", ascending=False).iloc[0]

    model_name = best_row["Model"]
    emb = int(best_row["Embedding Dim"])
    hid = int(best_row["Hidden Dim"])

    if model_name == "LSTM":
        model = LSTMClassifier(tokenizer.vocab_size, emb, hid, 2, tokenizer.pad_token_id)
    else:
        model = BiLSTMClassifier(tokenizer.vocab_size, emb, hid, 2, tokenizer.pad_token_id)

    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()

    evaluate_and_plot(model, val_loader, DEVICE, tag="Validation")

    # ROC & PR 曲线
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["input_ids"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            all_probs += probs[:, 1].cpu().tolist()
            all_labels += y.cpu().tolist()

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve on Validation Set")
    plt.legend()
    plt.savefig(os.path.join(output_root, "roc_curve_val.png"))
    plt.close()

    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve on Validation Set")
    plt.legend()
    plt.savefig(os.path.join(output_root, "pr_curve_val.png"))
    plt.close()

    print("✅ Validation ROC & PR curves have been generated.")

if __name__ == "__main__":
    plot_nested_by_fixed_param("lr")
    plot_nested_by_fixed_param("bs")
    evaluate_best_model_on_validation()
