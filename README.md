# Nepali MultiModal
## Introduction
Nepali Multimodal Large Language Model (N-MM-LLM) focuses on learning aligned representation between Nepali text and images. We utilize pre-trained NepBERTa for Nepali text understanding and CLIP's vision encoder for image processing, connecting them through trainable projection layers that create a shared embedding space. The framework employs contrastive learning to align the text and image embeddings. We would like to demonstrate that this approach effectively learns cross-modal relationships while preserving the strong pre-trained representations of both modalities. Rather than fine-tuning the entire models, we introduce an efficient approach that freezes the base encoders and only trains lightweight projection layers, significantly reducing computational requirements while maintaining performance. We will use an adaptive layer to map embeddings and then use an attention mechanism to learn textual relationships. This work will be the first step towards multimodal AI systems for Nepali language, providing a strong foundation for future research and applications in multimodal learning.


---

## Requirements  
- **Python**: 3.7+  
- **PyTorch**: 1.8.0+  
- **CUDA**: 9.2 or higher  
- **Key Libraries**:  
  - `timm`  
  - `transformers`  
  - `albumentations`  
  - `tqdm`  

---


## Installation  

### 1. Create Conda Environment
```bash  
conda create --name multimodal_llm python=3.8  
conda activate multimodal_llm 
```

### 2 Install PyTorch with CUDA 9.2
```bash  
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=9.2 -c pytorch  
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt  
```

## Dataset Preparation
1. Download Flickr8k Dataset
Organize the dataset as:
datasets/           
├── captions.csv      
└── translated_nepali_captions.txt

2. Preprocess Data
```bash
python src/multi_model_embedding_fusion/utils.py
```

3. Configuration
Modify src/multi_model_embedding_fusion/config.py to adjust:
* Model hyperparameters
* Dataset paths
* Training settings

## Training 
1. Train Fusion Model
```bash
python src/multi_model_embedding_fusion/trainer.py
```
2. Train Text Generation Model
```bash
python src/multi_model_text_generation/trainer.py
```

## Repository Structure
Nepali-MultiModal/
├── src/
│   ├── multi_model_embedding_fusion/
│   │   ├── data/
│   │   ├── datasets/
│   │   │   ├── captions.csv
│   │   │   └── translated_nepali_captions.txt
│   │   ├── models/
│   │   │   ├── model.py
│   │   │   └── multimodal_fusion.py
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── Nepali_MultiModal.ipynb
│   │   ├── setup_colab.py
│   │   ├── trainer.py
│   │   └── utils.py
│   │
│   ├── multi_model_text_generation/
│   │   ├── data/
│   │   ├── models/
│   │   │   ├── get_embeddings.py
│   │   │   ├── layers.py
│   │   │   ├── multi_head_attention.py
│   │   │   ├── positional_embedding.py
│   │   │   ├── transformer_block.py
│   │   │   └── transformer.py
│   │   ├── config.py
│   │   ├── main.py
│   │   └── trainer.py
│
├── LICENSE
├── requirements.txt
└── README.md
                       


