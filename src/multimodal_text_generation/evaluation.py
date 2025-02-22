from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from indicnlp.tokenize import indic_tokenize
import unicodedata
from transformers import  AutoTokenizer,AutoModel
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


tokenizer = AutoTokenizer.from_pretrained("NepBERTa/NepBERTa")
text_model = AutoModel.from_pretrained("NepBERTa/NepBERTa",from_tf=True)

image_model = models.resnet50(pretrained=True)
image_model.fc = nn.Linear(image_model.fc.in_features, 768)  
image_model.eval()


def indic_tokenizer(text):
    return unicodedata.normalize('NFC', text).strip().lower()


def calculate_bleu_score(references, generated):
   
    generated = indic_tokenizer(generated)
    tokenized_candidate = indic_tokenize.trivial_tokenize(generated)

    references = [indic_tokenizer(ref) for ref in references]
    tokenized_references = [indic_tokenize.trivial_tokenize(ref) for ref in references]
   
    smoothing_function = SmoothingFunction().method4  

    bleu_scores = {}

    for ref_text, ref_tokens in zip(references, tokenized_references):
        bleu_scores[ref_text] = {
            'bleu1': round(sentence_bleu([ref_tokens], tokenized_candidate, weights=(1, 0, 0, 0), smoothing_function=smoothing_function), 4),
            'bleu2': round(sentence_bleu([ref_tokens], tokenized_candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function), 4),
            'bleu3': round(sentence_bleu([ref_tokens], tokenized_candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function), 4),
            'bleu4': round(sentence_bleu([ref_tokens], tokenized_candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function), 4)
        }

    return bleu_scores



def nepberta_tokenizer(text):
    tokensss = tokenizer.tokenize(text)
    return tokensss
   
def calculate_rouge_score(references, generated):
    generated_normalized = generated #nepberta_tokenizer(generated)
    references_normalized = references #[nepberta_tokenizer(ref) for ref in references]

    print("Generated:", generated_normalized)
    print("References:", references_normalized)
   
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
   
    max_scores = {key: {'precision': 0, 'recall': 0, 'fmeasure': 0} for key in ['rouge1', 'rouge2', 'rougeL']}
   
    for reference in references_normalized:
        score = scorer.score(reference, generated_normalized)
        for key in max_scores:
            if score[key].fmeasure > max_scores[key]['fmeasure']:
                max_scores[key] = {
                    'precision': round(score[key].precision, 4),
                    'recall': round(score[key].recall, 4),
                    'fmeasure': round(score[key].fmeasure, 4)
                }
   
    return {'ROUGE Scores': max_scores}



transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# cmra = Cross Modal Retrieval Accuracy
def cmra(generated, input_image_path, references):

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
    
    return {
        "Recall@1": recall_at_1,
        "Recall@5": recall_at_5,
        "MRR": mrr,
        "Generated Caption Rank": rank
    }

