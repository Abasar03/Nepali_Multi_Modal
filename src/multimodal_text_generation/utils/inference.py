import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from src.multimodal_text_generation.models.transformer import Transformer

def generate_caption(model, tokenizer, fused_embedding, device, max_length=50):
    model.eval() 
    with torch.no_grad(): 
        fused_embedding = fused_embedding.view(1, 1024).to(device)  
        generated_ids = [tokenizer.cls_token_id]
        sep_token_id = torch.tensor(tokenizer.sep_token_id).to(device) 

        for _ in range(max_length): 
            inputs = torch.tensor([generated_ids]).to(device)
            logits = model(fused_embedding, inputs)[:, -1, :] 
            
            logits = logits / 0.5  
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            
            
            for token in generated_ids[-3:]:
                logits[0, token] *= 0.7
                 
            top_k = torch.topk(logits, 50)
            probs = torch.softmax(top_k.values, dim=-1)
            sampled_idx = torch.multinomial(probs, 1)
            next_token = top_k.indices.gather(1, sampled_idx)

            if torch.equal(next_token, sep_token_id) and len(generated_ids) > 5:
                break
                
            
            if len(generated_ids) >= 3 and next_token.item() in generated_ids[-3:]:
                continue
                
            generated_ids.append(next_token.cpu().item())

        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return decoded.capitalize().replace(" ##", "")  

def load_model(load_path, device):
    checkpoint = torch.load(load_path, map_location='cpu',weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained('NepBERTa/NepBERTa')
    model = Transformer(tokenizer)
    filtered_state_dict = {
        k: v for k, v in checkpoint.items()
        if k in model.state_dict()
    }
    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, tokenizer


def run_inference(model_path, test_image_embedding, device, max_attempts=3):
    try:
        print("Loading model...") 
        model, tokenizer = load_model(model_path, device)

        for attempt in range(max_attempts):
            print(f"\nAttempt {attempt + 1}/{max_attempts}")
            generated_caption = generate_caption(
                model,
                tokenizer,
                test_image_embedding,
                device
            )

            if generated_caption:
                return generated_caption

        print(f"Failed to generate caption after {max_attempts} attempts")
        return None 

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


 
 
 
