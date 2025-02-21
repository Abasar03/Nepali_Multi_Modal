from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from indicnlp.tokenize import indic_tokenize
import unicodedata
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import numpy as np
import torch
import clip
from PIL import Image


tokenizer = AutoTokenizer.from_pretrained("NepBERTa/NepBERTa")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)




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








def cmra(generated,input_image_path,references):
   
    image = Image.open(input_image_path)
    all_texts = references + [generated]
    inputs = processor(text=all_texts, images=[image], return_tensors="pt", padding=True)
   
    with torch.no_grad():
        image_embedding = model.get_image_features(inputs["pixel_values"])
        text_embeddings = model.get_text_features(inputs["input_ids"], inputs["attention_mask"])


    image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)


    similarities = (image_embedding @ text_embeddings.T).squeeze(0).cpu().numpy()


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

