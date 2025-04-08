import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import seaborn as sns

from data_loader import load_and_prepare
from lstm_model import LSTMClassifier
from bilstm_model import BiLSTMClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "outputs/Results/model/best_model.pth"
EVAL_OUTPUT = "outputs/Evaluation"
os.makedirs(EVAL_OUTPUT, exist_ok=True)

def evaluate_on_test():
    print("🔍 Evaluating the test set...")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_and_prepare()

    dataset = dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=128), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'label'])

    test_loader = DataLoader(dataset['test'], batch_size=32)

    result_df = pd.read_csv("outputs/Results/log/model_comparison_results.csv")
    best = result_df.sort_values("Validation F1", ascending=False).iloc[0]
    model_name = best["Model"]
    emb = int(best["Embedding Dim"])
    hid = int(best["Hidden Dim"])

    print(f"✅ Loading the best model：{model_name} (emb={emb}, hid={hid})")

    if model_name == "LSTM":
        model = LSTMClassifier(tokenizer.vocab_size, emb, hid, 2, tokenizer.pad_token_id)
    else:
        model = BiLSTMClassifier(tokenizer.vocab_size, emb, hid, 2, tokenizer.pad_token_id)

    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
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

    print(f"\n🎯 Test set results：Accuracy = {acc:.4f} | F1 Score = {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_OUTPUT, "confusion_matrix_test.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve on Test Set")
    plt.legend()
    plt.savefig(os.path.join(EVAL_OUTPUT, "roc_curve_test.png"))
    plt.close()

    # PR Curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve on Test Set")
    plt.legend()
    plt.savefig(os.path.join(EVAL_OUTPUT, "pr_curve_test.png"))
    plt.close()

    print("✅ Test set evaluation plots have been saved：ROC & PR Curve")

if __name__ == "__main__":
    evaluate_on_test()
