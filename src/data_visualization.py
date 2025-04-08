import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from data_loader import load_and_prepare

output_dir = 'outputs/visualizations'
os.makedirs(output_dir, exist_ok=True)

def visualize_data():
    dataset = load_and_prepare()
    visualize_data_distribution(dataset)
    for split in ['train', 'validation', 'test']:
        generate_wordcloud(dataset[split], split)

def visualize_data_distribution(dataset):
    for split in ['train', 'validation', 'test']:
        df = pd.DataFrame(dataset[split])
        sns.countplot(x='label', data=df)
        plt.title(f'{split} - Label Distribution')
        plt.xlabel('Label (0 = Real, 1 = Fake)')
        plt.ylabel('Count')
        plt.savefig(f'{output_dir}/{split}_distribution.png')
        plt.clf()

def generate_wordcloud(dataset, split_name):
    text = " ".join(dataset['text'])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    wordcloud.to_file(f'{output_dir}/{split_name}_wordcloud.png')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    visualize_data()
