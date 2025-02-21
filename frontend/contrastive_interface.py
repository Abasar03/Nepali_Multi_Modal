import os
import streamlit as st
import torch
from PIL import Image
from torch import nn
import timm
from transformers import AutoModel
from transformers import AutoTokenizer
from torchvision import transforms
import torch.nn.functional as F
import pandas as pd
from src.multimodal_embedding_fusion.models.model import ContrastiveModel
from src.multimodal_embedding_fusion.config import Configuration
# Initialize your model (ensure it is loaded correctly)
@st.cache_resource
def load_model():
    model = ContrastiveModel()
    model_path = r"C:\Users\riwas\Downloads\boosted_contrastive_model.pt"  # Use raw string to avoid backslash issues
    
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)  # strict=False allows partial loading

    model.to(torch.device("cpu"))
    model.eval()
    return model

# Function to calculate similarity
def get_similarity(image, text):
    # Load model once and store it in cache
    model = load_model()
    #preproscessing, simiilar to what we did for CM infrerence

    tokenizer = AutoTokenizer.from_pretrained("NepBERTa/NepBERTa")
    text_inputs = tokenizer(text, return_tensors = "pt", padding= True, truncation=True)

    #image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)
    # Extract features from the encoders
    text_inputs.pop("token_type_ids", None)
    text_features = model.text_encoder(**text_inputs)
    image_features = model.image_encoder(image_tensor)

    # Project features into the shared space
    text_embedding = model.text_projection(text_features)
    image_embedding = model.image_projection(image_features)  # Replace with actual method
    
    # Calculate similarity (cosine similarity or any other method you use)
    similarity_score = torch.cosine_similarity(image_embedding, text_embedding)
    
    return similarity_score.item()


# Streamlit interface
if __name__ == "__main__":
    option = st.selectbox("Choose an option", ["Similarity Check", "Caption Retrieval", "Image Retrieval"])
    st.write("You selected:", option)

    if option == "Similarity Check" or option == "Caption Retrieval":
        if option == "Similarity Check":
            st.title("Image-Text Similarity Checker")
            st.write("Upload an image different captions to check similarity scores.")
        elif option == "Caption Retrieval":
            st.title("Caption Retrieval")
            st.write("Upload an image and our model will generate a caption.")
        # Image uploader
        
        if option == "Similarity Check":
            image_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
            if image_file is not None:
                # Display the uploaded image
                image = Image.open(image_file).convert("RGB") #requires Image class from PIL
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Get 5 captions as input
                captions = []
                for i in range(5):
                    caption = st.text_input(f"Caption {i + 1}")
                    captions.append(caption)
                
                #check here to see if further changes on respective caption boxes represent respective index in similarity_scores
                
                if all(captions):

                    #all will check if all the text_input fields are filled
                    #such that we can move to calcualte the similarity
                    
                    scores = []
                    for caption in captions:
                        score = get_similarity(image, caption)
                        scores.append(score)
                    
                    # Display the similarity scores
                    for i, score in enumerate(scores):
                        st.write(f"Similarity score for Caption {i + 1}: {score:.4f}")
        
        elif option == "Caption Retrieval":
            image_file2 = st.file_uploader("Upload an Image for caption retrieval", type=["jpg", "png", "jpeg"])           
            if image_file2 is not None:
                # Display the uploaded image
                image = Image.open(image_file2).convert("RGB") #requires Image from PIL
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                #Retrieve captions from captions.csv
                df = pd.read_csv("src/multimodal_embedding_fusion/datasets/captions.csv")
                captions = df['caption'].head(100).tolist()
                st.write("The test captions: ")
                st.write(captions)
                
                st.write("The scores calculated for each caption: ")
                scores = []
                for caption in captions:
                    score = get_similarity(image, caption)
                    scores.append(score)
                
                captions_with_scores = [(caption, score) for caption, score in zip(captions, scores)]

                # Sort the list in descending order by score
                captions_with_scores.sort(key=lambda x: x[1], reverse=True)

                # Get the top 3 captions
                top_3_captions = captions_with_scores[:3]

                # Display the top 3 captions
                for i, (caption, score) in enumerate(top_3_captions):
                    st.write(f"Top {i+1}: {caption} - {score:.4f}")

                #st.write(f"** Retrived caption: ** {final_caption}")
                st.balloons()
    elif option == "Image Retrieval":
        image_inputs = []
        csv_path = "src/multimodal_embedding_fusion/datasets/captions.csv"  # Update with actual CSV path
        image_dir = r"C:\Users\riwas\Downloads\datasets-20250216T212255Z-001\datasets\ficker8k_images"

        df = pd.read_csv(csv_path)

        # Select every 6th image and take only the first 10 such selections
        total_samples = 100
        selected_images = df.iloc[::6].head(total_samples)

        st.title("Image retrieval")

        for _, row in selected_images.iterrows():
            image_path = os.path.join(image_dir, row['image'])  # Assuming column name is 'image_name'
            
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")  # Ensure it's RGB
                #st.image(image)
                image_inputs.append(image)  # Store for encoder
            
            else:
                st.write(f"Image not found: {row['image']}")

        text = st.text_input("insert query to retrieve image...")

        similarity_scores = []

        if text:  # Runs only when text is provided
            st.write("### Similarity Scores:")
            
            for i in range(len(image_inputs)):  
                similarity = get_similarity(image_inputs[i], text)  
                similarity_scores.append(similarity)  # Store scores
                print(f"Similarity for Image {i+1}: {similarity:.4f}")
            
            # Display scores in Streamlit
            for i, score in enumerate(similarity_scores):
                st.write(f"Image {i+1}: {score:.4f}")
            max_index = max(enumerate(similarity_scores), key=lambda x: x[1])[0]
            retrieved_image = image_inputs[max_index] 
            st.image(retrieved_image, caption = f"The retrieved image out of {total_samples} based on caption:{text}" )
            st.balloons()  # Celebration effect in Streamlit