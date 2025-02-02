import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import kagglehub
import re
from collections import Counter, defaultdict
import albumentations as A
from src.multimodal_embedding_fusion.config import Configuration

ANNOTATIONS_PATH = '/root/.cache/kagglehub/datasets/bipeshrajsubedi/flickr8k-nepali-dataset/versions/1/translated_nepali_captions.txt'
IMAGE_DIR = '/root/.cache/kagglehub/datasets/bipeshrajsubedi/flickr8k-nepali-dataset/versions/1/ficker8k_images/ficker8k_images'
PROCESSED_IMAGE_DIR = '/root/.cache/kagglehub/datasets/processed_images'
CLEANED_ANNOTATIONS_PATH = '/root/.cache/kagglehub/datasets/bipeshrajsubedi/flickr8k-nepali-dataset/versions/1/cleaned_translated_nepali_captions.txt'

# Nepali stopwords
NEPALI_STOPWORDS = set([
    'र', 'को', 'मा', 'छ', 'एक', 'ले', 'का', 'तथा', 'साथ', 'यहाँ', 'हुन', 'तिमी', 'हामी', 'यो', 'त्यो', 'जसले',
    'यो', 'त्यो', 'छैन', 'गर्दै', 'सक्ने', 'चाहिँ', 'कसैले', 'कसरी', 'पनि', 'हुनसक्छ', 'भएको', 'आफ्नो', 'का', 'छ'
])

# Utility Functions
def get_lr(optimizer):
    """Get the current learning rate from the optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_transforms(mode='train'):
    """Get image transformations for training or validation."""
    return A.Compose([
        A.Resize(Configuration.size, Configuration.size, always_apply=True),
        A.Normalize(max_pixel_value=255.0, always_apply=True),
    ])

def make_train_valid_dfs():
    """Split the dataset into training and validation sets."""
    dataframe = pd.read_csv("captions.csv")
    max_id = dataframe["id"].max() + 1 if not Configuration.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(image_ids, size=int(0.2 * len(image_ids)), replace=False)
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe

def regex_tokenizer(text):
    """Tokenize text using a regex pattern for Nepali words."""
    pattern = r'[ऀ-ॿ]+'  # Regex pattern to match only Devanagari words
    return re.findall(pattern, text)

def check_stopwords(caption, stopwords):
    """Check for stopwords in a caption and return them."""
    tokens = regex_tokenizer(caption)
    return [word for word in tokens if word in stopwords]

def calculate_brightness(image_path):
    """Calculate the brightness of an image."""
    try:
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        np_image = np.array(image)
        return np.mean(np_image)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_and_save_image(image_name):
    """Resize and save an image."""
    img_path = os.path.join(IMAGE_DIR, image_name)
    try:
        with Image.open(img_path) as img:
            img_resized = img.resize((224, 224))  # Resize to 224x224 pixels
            new_image_name = f"resized_{image_name}"
            new_image_path = os.path.join(PROCESSED_IMAGE_DIR, new_image_name)
            img_resized.save(new_image_path)
            return new_image_name, img_resized.size[0], img_resized.size[1]
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        return None, None, None

# Dataset Preparation
def load_and_preprocess_annotations():
    """Load and preprocess the annotations file."""
    with open(ANNOTATIONS_PATH, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        image_caption_split = line.strip().split('#')
        image_name = image_caption_split[0]
        caption = '#'.join(image_caption_split[1:]).split(' ', 1)[1]
        data.append((image_name, caption))
    return pd.DataFrame(data, columns=['image_name', 'caption'])

def clean_annotations_file():
    """Clean the annotations file by handling multiple '#' symbols."""
    with open(ANNOTATIONS_PATH, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    cleaned_lines = []
    for line in lines:
        if line.count('#') > 1:
            parts = line.split('#', 2)
            cleaned_line = f"{parts[0]}#{parts[1]}#{parts[2].strip()}"
        else:
            cleaned_line = line.strip()
        cleaned_lines.append(cleaned_line)

    with open(CLEANED_ANNOTATIONS_PATH, 'w', encoding='utf-8') as output_file:
        output_file.writelines("\n".join(cleaned_lines))

# Data Analysis
def analyze_captions(annotations_df):
    """Analyze captions for word counts, unique words, and stopwords."""
    vocab = Counter()
    grouped_by_word_count = defaultdict(list)
    unique_words_per_caption = []

    for text in annotations_df['caption']:
        tokens = regex_tokenizer(text)
        vocab.update(tokens)
        total_words = len(tokens)
        grouped_by_word_count[total_words].append(text)
        unique_words_per_caption.append(len(set(tokens)))

    max_unique_words = max(unique_words_per_caption)
    index_of_max = unique_words_per_caption.index(max_unique_words)
    caption_with_most_unique_words = annotations_df.iloc[index_of_max]['caption']

    print(f"Maximum number of unique words in a single caption: {max_unique_words}")
    print(f"Caption with the most unique words: {caption_with_most_unique_words}")

    # Count stopwords
    all_stopwords = []
    for caption in annotations_df['caption']:
        all_stopwords.extend(check_stopwords(caption, NEPALI_STOPWORDS))
    stopword_counts = Counter(all_stopwords)
    print("Most common stopwords:", stopword_counts.most_common())

def analyze_images(image_dir):
    """Analyze image metadata (width, height, aspect ratio)."""
    image_data = []
    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff')):
            try:
                with Image.open(os.path.join(image_dir, image_name)) as img:
                    width, height = img.size
                    aspect_ratio = width / height
                    image_data.append({
                        'filename': image_name,
                        'width': width,
                        'height': height,
                        'aspect_ratio': aspect_ratio
                    })
            except Exception as e:
                print(f"Error processing {image_name}: {e}")

    df = pd.DataFrame(image_data)
    print("Image File Types Distribution:")
    print(df['file_type'].value_counts())

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df['width'], bins=20, color='blue', alpha=0.7)
    plt.title('Distribution of Image Widths')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(df['height'], bins=20, color='green', alpha=0.7)
    plt.title('Distribution of Image Heights')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Load and preprocess annotations
    annotations_df = load_and_preprocess_annotations()

    # Clean annotations file
    clean_annotations_file()

    # Analyze captions
    analyze_captions(annotations_df)

    # Analyze images
    analyze_images(IMAGE_DIR)

    # Process and save resized images
    os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
    image_metadata = []
    for image_name in tqdm(os.listdir(IMAGE_DIR), desc="Processing images"):
        new_image_name, width, height = process_and_save_image(image_name)
        if new_image_name:
            image_metadata.append({
                'filename': new_image_name,
                'width': width,
                'height': height
            })

    # Save image metadata to a DataFrame
    image_metadata_df = pd.DataFrame(image_metadata)
    print(image_metadata_df.head())