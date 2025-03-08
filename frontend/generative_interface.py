import streamlit as st
import base64
from io import BytesIO
from PIL import Image
from st_multimodal_chatinput import multimodal_chatinput

import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import dataset,DataLoader

from src.multimodal_text_generation.config import config
from src.multimodal_text_generation.models.transformer import Transformer

from src.multimodal_embedding_fusion.models.model import ContrastiveModel
from src.multimodal_embedding_fusion.models.multimodal_fusion import MultiModalFusion


from torchvision import transforms

@st.cache_resource
def load_models():
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    contrastive_model = ContrastiveModel()
    contrastive_model_path = r"~\Downloads\models\contrastive_model_new.pt"
    contrastive_model.load_state_dict(torch.load(contrastive_model_path, map_location="cpu"), strict= True)
    contrastive_model.eval()

    fusion_model = MultiModalFusion()
    fusion_model_path = r"~\Downloads\models\Downloads\small_fused_v2.pt"
    fusion_model.load_state_dict(torch.load(fusion_model_path, map_location="cpu"))
    fusion_model.eval()

    transformer_model = Transformer(tokenizer)
    transformer_model_path = r"~\Downloads\models\Downloads\small_autoregressive_v2.pt"

    transformer_model.load_state_dict(torch.load(transformer_model_path, map_location="cpu"), strict= True)
    transformer_model.eval()

    return contrastive_model, fusion_model, transformer_model



def Pipeline_test(input_image=None, input_text=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('NepBERTa/NepBERTa')

    contrastive_model, fusion_model, transformer_model = load_models()

    with torch.no_grad():
        image_projected = None
        text_projected = None
        
        if input_image is not None:
            image_features = contrastive_model.image_encoder(input_image)
            image_projected = contrastive_model.image_projection(image_features)
            
        if input_text is not None:
            text_features = contrastive_model.text_encoder(
                input_ids=input_text['input_ids'],
                attention_mask=input_text['attention_mask']
            )
            text_projected = contrastive_model.text_projection(text_features)

        if image_projected is not None and text_projected is not None:
            fused_embedding = fusion_model(image_projected, text_projected)
        elif image_projected is not None:
            fused_embedding = fusion_model(image_projection=image_projected)
        elif text_projected is not None:
            fused_embedding = fusion_model(text_projection=text_projected)
        else:
            raise ValueError('Must provide at least one input (image or text).')


        if len(fused_embedding.shape) == 3:
            fused_embedding = fused_embedding.squeeze(1)

        if fused_embedding.shape[-1] != 1024:
            if fused_embedding.shape[-1] < 1024:
                padding = torch.zeros(
                    fused_embedding.size(0),
                    1024 - fused_embedding.shape[-1]
                ).to(device)
                fused_embedding = torch.cat([fused_embedding, padding], dim=-1)
            else:
                fused_embedding = fused_embedding[:, :1024]

        print(f"Post-padding shape: {fused_embedding.shape}")

        input_ids = torch.tensor([tokenizer.cls_token_id]).unsqueeze(0).to(device)

        for _ in range(config.max_seq_len - 1):
            outputs = transformer_model(fused_embedding, input_ids)
            next_token = outputs.argmax(-1)[:, -1].unsqueeze(-1) 
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.sep_token_id:
                break

        generated_caption = tokenizer.decode(
            input_ids.squeeze().tolist(),
            skip_special_tokens=True
        )

        return generated_caption

def decode_base64(img_url):
    if img_url.startswith('data:image'):
        img_url = img_url.split(',')[1]
    img_data = base64.b64decode(img_url)
    image = Image.open(BytesIO(img_data))
    return image

with st.container():
    chatinput = multimodal_chatinput()

    images = []
    messages = []
    image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    tokenizer = AutoTokenizer.from_pretrained('NepBERTa/NepBERTa')
    if(chatinput):
        st.write(chatinput)
        if chatinput['text'] and chatinput['images']:
            #handle both
            text = chatinput['text']
            uploaded_image = chatinput['images']
            image = decode_base64(uploaded_image[0])   
            st.write(image)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            processed_image = image_transform(image).unsqueeze(0).to(device)
            text_input = tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=128,
                truncation=True
            ).to(device)

            output = Pipeline_test(input_image=processed_image, input_text=text_input)
            st.write(f'generated caption: {output}')



        elif chatinput['text']:
            text = chatinput['text']
            #handle the text generation
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            text_input = tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=128,
                truncation=True
            ).to(device)

            output = Pipeline_test(input_text=text_input)
            st.write(f'generated caption: {output}')

        
        elif chatinput['images']:


            img_url = chatinput['images'] 
            image = decode_base64(img_url[0])
            st.write(image)

            #handle image
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            processed_image = image_transform(image).unsqueeze(0).to(device)

            output = Pipeline_test(input_image=processed_image)
            st.write(f'generated caption: {output}')
        else:
            st.write('The weather seems great huh!')