import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer
from src.multimodal_embedding_fusion.config import Configuration
from src.multimodal_embedding_fusion.models.model import ProjectionHead,ContrastiveModel
from src.multimodal_embedding_fusion.data.dataset import build_loaders
from src.multimodal_embedding_fusion.utils import make_train_valid_dfs

import torch.nn.functional as F
from tqdm.autonotebook import tqdm
from torch.utils.data import Subset, DataLoader

import torch.nn as nn
from transformers import AutoTokenizer
from PIL import Image
from torchvision import transforms

class MultiModalFusion(nn.Module): 
    def __init__(self,
                 image_embedding=Configuration.image_embedding,
                 text_embedding=Configuration.text_embedding,
                 fusion_dim=Configuration.fusion_dim):
        super().__init__()
        self.fusion_dim = fusion_dim

        self.image_input_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.5),  
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        self.text_input_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.5), 
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        
        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=8,
                dropout=0.5
            ) for _ in range(3)  
        ])

        self.attention_weights = nn.Parameter(torch.ones(3) / 3) 

       
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )

        self.semantic_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(0.5)
            ) for _ in range(2)
        ])

        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),  
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim * 2, fusion_dim * 2)
        )

        
        self.orthogonal_weight = 0.1

    def orthogonal_regularization(self, features):
        features_norm = F.normalize(features, p=2, dim=-1)
        sim_matrix = torch.matmul(features_norm, features_norm.transpose(-2, -1))

        mask = torch.eye(sim_matrix.size(-1), device=sim_matrix.device)
        sim_matrix = sim_matrix * (1 - mask)

        ortho_loss = torch.mean(torch.abs(sim_matrix))
        return ortho_loss
    
    def forward(self, image_projection=None, text_projection=None, return_ortho_loss=False):
        if image_projection is not None and text_projection is not None:
            image_proj = self.image_input_proj(image_projection)
            text_proj = self.text_input_proj(text_projection)
            attention_outputs = []
            for i, attention in enumerate(self.attention_heads):
                att_out, _ = attention(
                    query=image_proj.unsqueeze(0),
                    key=text_proj.unsqueeze(0),
                    value=text_proj.unsqueeze(0)
                )
                attention_outputs.append(att_out.squeeze(0) * self.attention_weights[i])

            combined_attention = sum(attention_outputs)

            gate_values = self.gate(combined_attention)
            gated_features = combined_attention * gate_values

            semantic_features = [proj(gated_features) for proj in self.semantic_projections]

            combined_features = torch.cat([gated_features] + semantic_features, dim=-1)

            fused = self.final_fusion(combined_features)

            if return_ortho_loss:
                ortho_loss = self.orthogonal_regularization(fused)
                return fused, ortho_loss
            return fused
        else:
            input_tensor = image_projection if image_projection is not None else text_projection
            semantic_features = [proj(input_tensor) for proj in self.semantic_projections]
            combined_features = torch.cat([input_tensor] + semantic_features, dim=-1)
            return self.final_fusion(combined_features)
  
def train_combined(model_path, num_samples=None):
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = AutoTokenizer.from_pretrained(Configuration.text_tokenizer)

    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

    contrastive_model = ContrastiveModel().to(Configuration.device)
    contrastive_model.load_state_dict(torch.load(model_path))
    contrastive_model.eval()

    fusion_model = MultiModalFusion().to(Configuration.device)
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    reconstruction_criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience = 0
    max_patience = 5
    for epoch in range(17):
        fusion_model.train()
        train_loss = 0
        train_recon_loss = 0
        train_contrast_loss = 0
        train_ortho_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            batch = {k: v.to(Configuration.device) for k, v in batch.items() if k != 'caption'}

            with torch.no_grad():
                image_features = contrastive_model.image_encoder(batch['image'])
                text_features = contrastive_model.text_encoder(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                image_projected = contrastive_model.image_projection(image_features)
                text_projected = contrastive_model.text_projection(text_features)

            if fusion_model.training:
                noise_scale = 0.05 
                image_noise = torch.randn_like(image_projected) * noise_scale
                text_noise = torch.randn_like(text_projected) * noise_scale
                image_projected = image_projected + image_noise
                text_projected = text_projected + text_noise

            fused, ortho_loss = fusion_model(image_projected, text_projected, return_ortho_loss=True)

            target = torch.cat([image_projected, text_projected], dim=-1)
            recon_loss = 0.5 * reconstruction_criterion(fused, target)  

            contrast_loss = 0
            batch_size = fused.size(0)
            if batch_size > 1:
                for i in range(batch_size):
                    for j in range(i + 1, batch_size):
                        contrast_loss += contrastive_loss(
                            fused[i],
                            fused[j],
                            negative_margin=0.2
                        )
                contrast_loss = contrast_loss / (batch_size * (batch_size - 1) / 2)

            total_loss = (
                0.3 * recon_loss +  
                0.5 * contrast_loss +  
                0.2 * ortho_loss 
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()
            train_recon_loss += recon_loss.item()
            train_contrast_loss += contrast_loss.item() if batch_size > 1 else 0


        fusion_model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_contrast_loss = 0
        val_ortho_loss = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f'Epoch {epoch + 1} - Validation'):
                batch = {k: v.to(Configuration.device) for k, v in batch.items() if k != 'caption'}

                image_features = contrastive_model.image_encoder(batch['image'])
                text_features = contrastive_model.text_encoder(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                image_projected = contrastive_model.image_projection(image_features)
                text_projected = contrastive_model.text_projection(text_features)

                fused, ortho_loss = fusion_model(image_projected, text_projected, return_ortho_loss=True)
                target = torch.cat([image_projected, text_projected], dim=-1)
                recon_loss = 0.5 * reconstruction_criterion(fused, target)

                contrast_loss = 0
                batch_size = fused.size(0)
                if batch_size > 1:
                    for i in range(batch_size):
                        for j in range(i + 1, batch_size):
                            contrast_loss += contrastive_loss(
                                fused[i],
                                fused[j],
                                negative_margin=0.2
                            )
                    contrast_loss = contrast_loss / (batch_size * (batch_size - 1) / 2)

                total_loss = (
                    0.3 * recon_loss +
                    0.5 * contrast_loss +
                    0.2 * ortho_loss
                )

                val_loss += total_loss.item()
                val_recon_loss += recon_loss.item()
                val_contrast_loss += contrast_loss.item() if batch_size > 1 else 0
                val_ortho_loss += ortho_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_recon = train_recon_loss / len(train_loader)
        avg_train_contrast = train_contrast_loss / len(train_loader)
        avg_train_ortho = train_ortho_loss / len(train_loader)

        avg_val_loss = val_loss / len(valid_loader)
        avg_val_recon = val_recon_loss / len(valid_loader)
        avg_val_contrast = val_contrast_loss / len(valid_loader)
        avg_val_ortho = val_ortho_loss / len(valid_loader)

        print(f'\nEpoch {epoch + 1}:')
        print(f'Training - Total Loss: {avg_train_loss:.4f}, '
              f'Recon Loss: {avg_train_recon:.4f}, '
              f'Contrast Loss: {avg_train_contrast:.4f}, '
              f'Ortho Loss: {avg_train_ortho:.4f}')
        print(f'Validation - Total Loss: {avg_val_loss:.4f}, '
              f'Recon Loss: {avg_val_recon:.4f}, '
              f'Contrast Loss: {avg_val_contrast:.4f}, '
              f'Ortho Loss: {avg_val_ortho:.4f}')

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(fusion_model.state_dict(), 'best_fusion_model.pt')
            patience = 0
            print("Saved new best model!")
        else:
            patience += 1
            if patience >= max_patience:
                print("Early stopping triggered!")
                break

    print('Training completed!')
    return fusion_model    


# def train_combined(model_path):              
#     train_df, valid_df = make_train_valid_dfs()
#     tokenizer = AutoTokenizer.from_pretrained(Configuration.text_tokenizer)
#     train_loader = build_loaders(train_df, tokenizer, mode="train") 
#     valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
#     contrastive_model = ContrastiveModel().to(Configuration.device)
#     contrastive_model.load_state_dict(torch.load(model_path))
#     contrastive_model.eval()   
    
#     fusion_model = MultiModalFusion().to(Configuration.device)
#     optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=1e-4)
#     criterion = nn.MSELoss()    
    
#     best_loss = float('inf')  
    
#     for epoch in range(Configuration.num_epochs): 
#         fusion_model.train() 
#         train_loss = 0
        
#         for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
#             batch = {k: v.to(Configuration.device) for k, v in batch.items() if k != 'caption'}
            
#             with torch.no_grad():    
#                 image_features = contrastive_model.image_encoder(batch['image'])
#                 text_features = contrastive_model.text_encoder(
#                     input_ids=batch['input_ids'],
#                     attention_mask=batch['attention_mask'] 
#                 )

#             image_projected = contrastive_model.image_projection(image_features)
#             text_projected = contrastive_model.text_projection(text_features)
            
#             fused = fusion_model(image_projected, text_projected)
#             target = torch.cat([image_projected, text_projected], dim=-1) 

#             loss = criterion(fused, target)     
            
#             optimizer.zero_grad()       
#             loss.backward()
#             optimizer.step()
             
#             train_loss += loss.item()
        
#         avg_train_loss = train_loss / len(train_loader)
#         print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}')
        
#         if avg_train_loss < best_loss:
#             best_loss = avg_train_loss
#             torch.save(fusion_model.state_dict(), 'fused_embeddings.pt')
            
        
#         fusion_model.eval()
#         valid_loss = 0
        
#         with torch.no_grad():
#             for batch in tqdm(valid_loader, desc=f'Epoch {epoch + 1} - Validation'):
#                 batch = {k: v.to(Configuration.device) for k, v in batch.items() if k != 'caption'}
                
#                 image_features = contrastive_model.image_encoder(batch['image'])
#                 text_features = contrastive_model.text_encoder(
#                     input_ids=batch['input_ids'],
#                     attention_mask=batch['attention_mask']
#                 ) 
                
#                 image_projected = contrastive_model.image_projection(image_features)
#                 text_projected = contrastive_model.text_projection(text_features)
            
#                 fused = fusion_model(image_projected, text_projected)
#                 target = torch.cat([image_projected, text_projected], dim=-1)               
                
#                 loss = criterion(fused, target) 
#                 valid_loss += loss.item()
        
#         avg_valid_loss = valid_loss / len(valid_loader)
#         print(f'Epoch {epoch + 1}, Validation Loss: {avg_valid_loss:.4f}')
        
#         if avg_valid_loss < best_loss: 
#             best_loss = avg_valid_loss 
#             torch.save(fusion_model.state_dict(), 'fused_embeddings.pt')
#             print(f'Saved best model with Validation Loss: {best_loss:.4f}')

#     print('Training completed!') 
#     return fusion_model
    
    

