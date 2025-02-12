import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer
from src.multimodal_embedding_fusion.config import Configuration
from src.multimodal_embedding_fusion.models.model import ProjectionHead,ContrastiveModel
from src.multimodal_embedding_fusion.data.dataset import build_loaders
from src.multimodal_embedding_fusion.utils import make_train_valid_dfs


class MultiModalFusion(nn.Module): 
    def __init__( 
        self,
        image_embedding=Configuration.image_embedding,
        text_embedding=Configuration.text_embedding, 
        fusion_dim=Configuration.fusion_dim
    ):
        super().__init__()
        self.image_projection=ProjectionHead(embedding_dim=image_embedding)
        self.text_projection=ProjectionHead(embedding_dim=text_embedding)
        self.fusion_dim=fusion_dim  
        
        self.cross_attention=nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,    
            dropout=0.1
        )  
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),  
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
            
        )      
        
    def forward(self,image_features,text_features):
        image_projection=self.image_projection(image_features)
        text_projection=self.text_projection(text_features)

        if len(image_projection.shape) == 2:
            image_projection = image_projection.unsqueeze(0)
        if len(text_projection.shape) == 2:
            text_projection = text_projection.unsqueeze(0)
        
        fused,_=self.cross_attention(
            query=image_projection,
            key=text_projection,
            value=text_projection
        )
        return self.final_fusion(fused) 

def train_combined(model_path):    
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = AutoTokenizer.from_pretrained(Configuration.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    contrastive_model = ContrastiveModel().to(Configuration.device)
    contrastive_model.load_state_dict(torch.load(model_path))
    contrastive_model.eval() 
    
    fusion_model = MultiModalFusion().to(Configuration.device)
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()  
    
    best_loss = float('inf')
    
    for epoch in range(Configuration.num_epochs): 
        fusion_model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            batch = {k: v.to(Configuration.device) for k, v in batch.items() if k != 'caption'}
            
            with torch.no_grad():
                image_features = contrastive_model.image_encoder(batch['image'])
                text_features = contrastive_model.text_encoder(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
            
            image_projected = fusion_model.image_projection(image_features)
            text_projected = fusion_model.text_projection(text_features)
            
            if len(image_projected.shape) == 2:
                image_projected = image_projected.unsqueeze(0)
            if len(text_projected.shape) == 2:
                text_projected = text_projected.unsqueeze(0)
            
            fused = fusion_model.cross_attention(
                query=image_projected,
                key=text_projected,
                value=text_projected
            )[0]
            
            fused = fusion_model.final_fusion(fused)
            
            target = (image_projected + text_projected) / 2


            loss = criterion(fused, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}')
        
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(fusion_model.state_dict(), 'fused_embeddings.pt')
            
        
        fusion_model.eval()
        valid_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f'Epoch {epoch + 1} - Validation'):
                batch = {k: v.to(Configuration.device) for k, v in batch.items() if k != 'caption'}
                
                image_features = contrastive_model.image_encoder(batch['image'])
                text_features = contrastive_model.text_encoder(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                fused = fusion_model(image_features, text_features)
                target = (fusion_model.image_projection(image_features) + fusion_model.text_projection(text_features)) / 2
                loss = criterion(fused, target)
                valid_loss += loss.item()
        
        avg_valid_loss = valid_loss / len(valid_loader)
        print(f'Epoch {epoch + 1}, Validation Loss: {avg_valid_loss:.4f}')
        
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            torch.save(fusion_model.state_dict(), 'fused_embeddings.pt')
            print(f'Saved best model with Validation Loss: {best_loss:.4f}')

    print('Training completed!') 

