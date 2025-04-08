import os
import torch
import gradio as gr
import pandas as pd
from transformers import AutoTokenizer
from lstm_model import LSTMClassifier
from bilstm_model import BiLSTMClassifier

# Load best model parameters
best_df = pd.read_csv("outputs/Results/log/model_comparison_results.csv")
best = best_df.sort_values("Validation F1", ascending=False).iloc[0]

model_name = best["Model"]
emb = int(best["Embedding Dim"])
hid = int(best["Hidden Dim"])
dropout = 0.3  # fixed
model_path = "outputs/Results/model/best_model.pth"

# Initialize tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size

if model_name == "LSTM":
    model = LSTMClassifier(vocab_size, emb, hid, 2, tokenizer.pad_token_id)
else:
    model = BiLSTMClassifier(vocab_size, emb, hid, 2, tokenizer.pad_token_id)

model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

def predict(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs["input_ids"])
        probs_tensor = torch.softmax(logits, dim=1).squeeze()
        probs = probs_tensor.tolist()
        label = probs_tensor.argmax().item()

    result_text = "🟢 Real News" if label == 0 else "🔴 Fake News"
    return result_text, {
        "Real": round(probs[0], 2),
        "Fake": round(probs[1], 2)
    }


# Gradio web interface
gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=8, placeholder="Enter news title or content...", label="📝 News Text"),
    outputs=[
        gr.Textbox(label="📢 Prediction Result"),
        gr.Label(num_top_classes=2, label="📊 Prediction Probability")
    ],
    title="📰 Fake News Detector",
    description="✨ Enter any news content, and the model will automatically determine whether it is real or fake.\n✅ Currently using the best-performing model from training.",
    theme="soft",
    examples=[
        ["U.S. Democratic Senator Al Franken will announce his resignation on Thursday, a day after a majority of his Democratic Senate colleagues called for him to step down following a string of sexual misconduct allegations against him, CNN reported on Thursday, citing unnamed sources."],
        ["The FBI reportedly has found emails from Hillary Clinton s private server on the laptop computer seized from Anthony Weiner and they are not duplicates of those already found in the server probe, according to CBS News.It was not clear, however, if the emails were related in any way to the Clinton server scandal or how many new messages were found.Weiner s estranged wife, Huma Abedin who is Clinton s top aide has claimed she had no knowledge of the existence of any Clinton emails on her husband s laptop.The FBI on Monday obtained a warrant to look through the laptop s 650,000 emails. NYP"]
    ]
).launch()
