# 📰 Fake News Detection with LSTM & BiLSTM

This project implements a comprehensive **Fake News Detection** pipeline using classical deep learning architectures: **LSTM** and **BiLSTM**. It includes training, evaluation, visualization, and deployment via a Gradio-based web interface.

## 🚀 Highlights

- ✅ Fine-tuned LSTM & BiLSTM models on news datasets
- 📊 Visualization of training metrics and performance curves
- 🔎 Comparative experiments with 24 different hyperparameter combinations
- 🌐 Real-time fake news prediction via web app (Gradio)
- 📈 Evaluation using ROC, PR curves, and confusion matrices

---

## 📁 Project Structure

```plaintext
├── src/
│   ├── data_loader.py           # Load and preprocess dataset
│   ├── data_visualization.py    # Visualize class distribution & wordclouds
│   ├── lstm_model.py            # LSTM model definition
│   ├── bilstm_model.py          # BiLSTM model definition
│   ├── compare_models.py        # Train all 24 combinations & save logs/models
│   ├── evalCompare.py           # Plot training curves, eval best model on val set
│   ├── evalTest.py              # Evaluate best model on test set
│   ├── plot_summary.py          # F1 heatmap + radar chart
│   ├── utils.py                 # Evaluation helpers
│   └── app.py                   # Gradio web app for real-time inference
│
├── outputs/
│   ├── Results/
│   │   ├── log/                 # Training logs (.csv)
│   │   └── model/               # Saved models (.pth)
│   ├── Evaluation/
│   │   ├── heatmaps/            # F1 heatmaps
│   │   ├── radar/               # Radar comparison charts
│   │   ├── LSTM/...             # Metric curves (by model & structure)
│   │   └── BiLSTM/...
│   └── visualizations/          # Dataset distribution & wordclouds
