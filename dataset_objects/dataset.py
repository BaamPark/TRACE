import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import json
from sklearn.model_selection import KFold
    

class PPE_dataset(Dataset):
    def __init__(self, json_path, img_path, transform=None):
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        self.img_path = img_path
        self.json_data = json_data
        self.transform = transform

    def __getitem__(self, index):
        imagepath = self.json_data[index]['file']
        img = Image.open(f"{self.img_path}/{imagepath}")
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        attribute_label = self._encode_label(self.json_data[index])
        # attribute_label = torch.LongTensor(attribute_label)

        return img, attribute_label

    def __len__(self):
        return len(self.json_data)

    def _encode_label(self, json_query):
        ppe_map = {'gc':0, 'ga':1, 'gi':2, 'pr':3, 'gg':4, 'fc':5, 'ea':6, 'nc':7, 'rc':8, 'ma':9, 'hc':10, 'ha':11}
        onehot_enocding = torch.zeros(len(ppe_map))
        if json_query['gown'] != 'na':
            gown_code = ppe_map[json_query['gown']]
            onehot_enocding[gown_code] = 1

        if json_query['eyewear'] != 'na':
            eyewear_code = ppe_map[json_query['eyewear']]
            onehot_enocding[eyewear_code] = 1

        if json_query['mask'] != 'na':
            mask_code = ppe_map[json_query['mask']]
            onehot_enocding[mask_code] = 1

        if json_query['glove'] != 'na':
            glove_code = ppe_map[json_query['glove']]
            onehot_enocding[glove_code] = 1

        return onehot_enocding


class PPE_dataset_KFold(Dataset):
    def __init__(self, json_path, img_path, fold_idx=0, train_or_val='train', 
                 transform=None, random_state=42):
        """
        Args:
            json_path: Path to JSON file with labels
            img_path: Path to image directory
            fold_idx: Which fold to use (0 to k-1)
            train_or_val: 'train' or 'val' to specify split
            transform: Image transformations
            random_state: Random seed for reproducible splits
        """
        k=5
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        self.img_path = img_path
        self.transform = transform
        
        # Create k-fold splits
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        indices = np.arange(len(json_data))
        
        # Get the indices for this specific fold
        all_splits = list(kf.split(indices))
        train_idx, val_idx = all_splits[fold_idx]
        
        # Select indices based on train_or_val
        if train_or_val == 'train':
            selected_indices = train_idx
        elif train_or_val == 'val':
            selected_indices = val_idx
        else:
            raise ValueError("train_or_val must be 'train' or 'val'")
        
        # Filter json_data to only include selected indices
        self.json_data = [json_data[i] for i in selected_indices]
        self.indices = selected_indices

    def __getitem__(self, index):
        imagepath = self.json_data[index]['file']
        img = Image.open(f"{self.img_path}/{imagepath}")
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        attribute_label = self._encode_label(self.json_data[index])

        return img, attribute_label

    def __len__(self):
        return len(self.json_data)

    def _encode_label(self, json_query):
        ppe_map = {'gc':0, 'ga':1, 'gi':2, 'pr':3, 'gg':4, 'fc':5, 
                   'ea':6, 'nc':7, 'rc':8, 'ma':9, 'hc':10, 'ha':11}
        onehot_encoding = torch.zeros(len(ppe_map))
        
        if json_query['gown'] != 'na':
            gown_code = ppe_map[json_query['gown']]
            onehot_encoding[gown_code] = 1

        if json_query['eyewear'] != 'na':
            eyewear_code = ppe_map[json_query['eyewear']]
            onehot_encoding[eyewear_code] = 1

        if json_query['mask'] != 'na':
            mask_code = ppe_map[json_query['mask']]
            onehot_encoding[mask_code] = 1

        if json_query['glove'] != 'na':
            glove_code = ppe_map[json_query['glove']]
            onehot_encoding[glove_code] = 1

        return onehot_encoding


# if __name__ == "__main__":
#     # Example: 5-fold CV
#     k = 5
    
#     # Create datasets for fold 0
#     train_dataset = PPE_dataset_KFold(
#         json_path='data/pacn_dataset_mix/label.json',
#         img_path='data/pacn_dataset_mix/image',
#         fold_idx=0,
#         train_or_val='train'
#     )
    
#     val_dataset = PPE_dataset_KFold(
#         json_path='data/pacn_dataset_mix/label.json',
#         img_path='data/pacn_dataset_mix/image',
#         fold_idx=0,
#         train_or_val='val'
#     )
    
#     print(f"Fold 0 - Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
#     # Example: Loop through all folds
#     print("\nAll folds:")
#     for fold_idx in range(k):
#         train_dataset = PPE_dataset_KFold(
#             json_path='data/pacn_dataset_mix/label.json',
#             img_path='data/pacn_dataset_mix/image',
#             fold_idx=fold_idx,
#             train_or_val='train'
#         )
        
#         val_dataset = PPE_dataset_KFold(
#             json_path='data/pacn_dataset_mix/label.json',
#             img_path='data/pacn_dataset_mix/image',
#             fold_idx=fold_idx,
#             train_or_val='val'
#         )
        
#         print(f"Fold {fold_idx} - Train: {len(train_dataset)}, Val: {len(val_dataset)}")