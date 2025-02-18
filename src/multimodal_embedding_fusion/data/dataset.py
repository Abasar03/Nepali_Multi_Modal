import cv2
import torch
import torch.nn as nn
from src.multimodal_embedding_fusion.config import Configuration
from src.multimodal_embedding_fusion.utils import get_transforms


class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self,image_filenames,captions,tokenizer,transforms):
        self.image_filenames=image_filenames
        self.captions=list(captions)
        self.tokenizer=tokenizer
        self.transforms=transforms
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoded_captions=tokenizer(
            self.captions,
            padding='max_length',
            truncation=True,
            max_length=Configuration.max_length,
            return_tensors='np'
        )
        
        
    
    def __getitem__(self, idx):  
        # Process image  
        image_path = f"{Configuration.image_path}/{self.image_filenames[idx]}"
        image = cv2.imread(image_path) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=image)
        image_tensor = torch.tensor(augmented['image']).permute(2, 0, 1).float()

        # Get text components
        input_ids = torch.tensor(self.encoded_captions['input_ids'][idx])
        attention_mask = torch.tensor(self.encoded_captions['attention_mask'][idx])
        
        return {
            'image': image_tensor,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'caption': self.captions[idx]  
        }

    def __len__(self):
        return len(self.captions)   


def collate_fn(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = torch.stack([item['image'].to(Configuration.device) for item in batch])
    input_ids = torch.stack([item['input_ids'].to(Configuration.device) for item in batch])
    attention_masks = torch.stack([item['attention_mask'].to(Configuration.device) for item in batch])
    captions = [item['caption'] for item in batch]

    return {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'caption': captions
    }
    
def build_loaders(dataframe,tokenizer,mode):
    transforms=get_transforms(mode=mode)
    dataset=ImageTextDataset(
        dataframe['image'].values,
        dataframe['caption'].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader=torch.utils.data.DataLoader(
        dataset,
        batch_size=Configuration.batch_size,
        num_workers=Configuration.num_workers,
        shuffle=True if mode=='train' else False,
        collate_fn=collate_fn
    )
    return dataloader


