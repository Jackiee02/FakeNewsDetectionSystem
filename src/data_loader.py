# src/data_loader.py

from datasets import load_dataset

def load_and_prepare():
    """
    Load the ErfanMoosaviMonazzah/fake-news-detection-dataset-English dataset,
    and combine the title and text fields into a single input text.
    Returns a HuggingFace DatasetDict containing train/validation/test splits.
    """
    dataset = load_dataset("ErfanMoosaviMonazzah/fake-news-detection-dataset-English")

    # Combine title and text fields into a complete input text
    dataset = dataset.map(lambda example: {
        'text': example['title'] + ' ' + example['text'],
        'label': example['label']
    })

    return dataset
