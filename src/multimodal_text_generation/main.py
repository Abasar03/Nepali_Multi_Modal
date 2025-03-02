import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import dataset,DataLoader

from src.multimodal_text_generation.config import config
from src.multimodal_text_generation.models.transformer import Transformer
from src.multimodal_text_generation.data.dataset import CaptionEmbeddingDataset, collate_fn
from src.multimodal_text_generation.utils.inference import run_inference 
from src.multimodal_text_generation.trainer import train_model 

def main(): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained('NepBERTa/NepBERTa')
    text_model = AutoModel.from_pretrained("NepBERTa/NepBERTa", from_tf=True)
    
    loaded_results = torch.load('/content/drive/MyDrive/MinorProject_Nepali_MultiModal_LLM/prefix_and_word.pt')
    
    dataset = CaptionEmbeddingDataset(loaded_results,tokenizer)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    model = Transformer(tokenizer)
    
    num_epochs = 5 
    train_model(model, train_loader,val_loader, num_epochs, device)
    return model
    
# if __name__ == "__main__":
#     model=main()
#     torch.save(model.state_dict(), '/content/drive/MyDrive/Minor_project/autoregressive_model.pt')

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_results = torch.load('/content/drive/MyDrive/Minor_project/aligned_caption_embeddings.pt', weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained('NepBERTa/NepBERTa')
    dataset = CaptionEmbeddingDataset(loaded_results,tokenizer)
    test_caption, test_embedding = dataset[5]
    fused_embedding = test_embedding.clone().detach().to(device)
    
    model_path = '/content/drive/MyDrive/Minor_project/autoregressive_gen_model.pt'
    generated_caption = run_inference(model_path, fused_embedding, device)
    
    print("Original:", test_caption)
    print("Generated:", generated_caption)





            
