from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from indicnlp.tokenize import indic_tokenize
import unicodedata
from transformers import  AutoTokenizer,AutoModel
import numpy as np
import torch
import torch.nn as nn
from torchvision import models,transforms
from PIL import Image

def calculate_bleu_score(references, generated):
    def indic_tokenizer(text):
        return unicodedata.normalize('NFC', text).strip().lower()
    
    # Normalization and tokenization
    generated_norm = indic_tokenizer(generated)
    tokenized_candidate = indic_tokenize.trivial_tokenize(generated_norm)

    # Normalization and tokenization
    normalized_refs = [indic_tokenizer(ref) for ref in references]
    tokenized_references = [indic_tokenize.trivial_tokenize(ref) for ref in normalized_refs]
   
    smoothing_function = SmoothingFunction().method4  

    bleu_scores = {}
    for ref_text, ref_tokens in zip(normalized_refs, tokenized_references):
        bleu_scores[ref_text] = {
            'bleu1': round(sentence_bleu([ref_tokens], tokenized_candidate, weights=(1, 0, 0, 0), smoothing_function=smoothing_function), 4),
            'bleu2': round(sentence_bleu([ref_tokens], tokenized_candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function), 4),
            'bleu3': round(sentence_bleu([ref_tokens], tokenized_candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function), 4),
            'bleu4': round(sentence_bleu([ref_tokens], tokenized_candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function), 4)
        }
    
    best_bleu = {}
    for key in ['bleu1', 'bleu2', 'bleu3', 'bleu4']:
        best_bleu[key] = max(bleu_scores[ref][key] for ref in bleu_scores)
    
    return best_bleu



def calculate_rouge_score(references, generated):
    
    def tokenize(text):
        try:
            return indic_tokenize.trivial_tokenize(text.strip())
        except:
            return []
    
    def get_ngram_counts(tokens, n):
        return {(tuple(tokens[i:i+n])): 1 for i in range(len(tokens)-n+1)} if len(tokens) >= n else {}
    
    def calculate_overlap(ref_counts, gen_counts):
        return sum(min(ref_counts.get(ngram, 0), gen_counts.get(ngram, 0)) for ngram in gen_counts)
    
    def rouge_n(ref_tokens, gen_tokens, n):
        ref_counts = get_ngram_counts(ref_tokens, n)
        gen_counts = get_ngram_counts(gen_tokens, n)
        
        overlap = calculate_overlap(ref_counts, gen_counts)
        ref_ngrams = max(1, len(ref_tokens) - n + 1)
        gen_ngrams = max(1, len(gen_tokens) - n + 1)
        
        precision = overlap / gen_ngrams
        recall = overlap / ref_ngrams
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "fmeasure": round(f1, 4)
        }
    
    def rouge_l(ref_tokens, gen_tokens):
        m, n = len(ref_tokens), len(gen_tokens)
        if m == 0 or n == 0:
            return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
        
        prev_row = [0] * (n + 1)
        for i in range(1, m + 1):
            current_row = [0] * (n + 1)
            for j in range(1, n + 1):
                if ref_tokens[i-1] == gen_tokens[j-1]:
                    current_row[j] = prev_row[j-1] + 1
                else:
                    current_row[j] = max(prev_row[j], current_row[j-1])
            prev_row = current_row
        
        lcs = prev_row[n]
        precision = lcs / n
        recall = lcs / m
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "fmeasure": round(f1, 4)
        }

    gen_tokens = tokenize(generated)
    refs_tokens = [tokenize(ref) for ref in references]

    best_scores = {
        "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
    }

    for ref_tokens in refs_tokens:
        r1 = rouge_n(ref_tokens, gen_tokens, 1)
        r2 = rouge_n(ref_tokens, gen_tokens, 2)
        rL = rouge_l(ref_tokens, gen_tokens)
        
        for metric in ["rouge1", "rouge2", "rougeL"]:
            current_score = locals()[f"r{metric[-1]}"]
            if current_score["fmeasure"] > best_scores[metric]["fmeasure"]:
                best_scores[metric] = current_score

    return {"ROUGE Scores": best_scores}




def calculate_cmra(generated, input_image_path, references):
    
    tokenizer = AutoTokenizer.from_pretrained("NepBERTa/NepBERTa")
    text_model = AutoModel.from_pretrained("NepBERTa/NepBERTa",from_tf=True)

    image_model = models.resnet50(pretrained=True)
    image_model.fc = nn.Linear(image_model.fc.in_features, 768)  
    image_model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(input_image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  
    
    all_texts = references + [generated]
    inputs = tokenizer(all_texts, padding=True, return_tensors="pt", truncation=True)
    
    # image embeddings
    with torch.no_grad():
        image_features = image_model(image_tensor)  
    
    # Normalization
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # text embeddings
    with torch.no_grad():
        text_outputs = text_model(**inputs)
        text_embeddings = text_outputs.last_hidden_state.mean(dim=1)
    
    # Normalization
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    
    similarities = torch.matmul(image_features, text_embeddings.T).squeeze(0).cpu().numpy()
    
    sorted_indices = np.argsort(-similarities)
    rank = np.where(sorted_indices == len(references))[0][0] + 1
    
    recall_at_1 = 1 if rank == 1 else 0
    recall_at_5 = 1 if rank <= 5 else 0
    mrr = 1 / rank
    generated_similarity = similarities[len(references)]

    return {
        "Recall@1": recall_at_1,
        "Recall@5": recall_at_5,
        "MRR": mrr,
        "Generated Caption Rank": rank,
        "Generated Caption Similarity": generated_similarity
    }