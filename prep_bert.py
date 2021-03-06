import torch
from torch import Tensor
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, sampler
from typing import Dict, List, Tuple, Optional, Union
from transformers import PreTrainedTokenizerBase
from sklearn.model_selection import train_test_split
import pandas as pd

class BertEncoder():
    """
    This class handles the data preprocessing step: tokenization in preparation for train test split and dataloader
    """

    def __init__(self, input_data: List[str], tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer
        self.input_data = input_data
    
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx]
    
    def tokenize(self, max_len: int) -> Tuple[Tensor, Tensor]:
        """
        This function tokenize the input data and return tokenized data

        Args:
            max_len ([int]): maxium sequance length.
        """
        encoded_dicts = []
        for text in self.input_data:
            encoded_dict = self.tokenizer(
                text = text,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=max_len,
                return_attention_mask=True,
                return_tensors='pt'
            )
            encoded_dicts.append(encoded_dict)
        input_ids = torch.cat([encoded_dict['input_ids'] for encoded_dict in encoded_dicts], dim=0)
        attention_masks = torch.cat([encoded_dict['attention_mask'] for encoded_dict in encoded_dicts], dim=0)

        return input_ids, attention_masks

def build_dataloaders(
    input_ids: Tensor,
    attention_masks: Tensor, 
    labels: Tensor,
    batch_size: Tuple[int, int], 
    train_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    This function takes in tokenized copus, and returns train and validation dataloaders.

    Args:
        input (List[Tensor]): input data containing input_ids, attention_masks
        batch_size ([int]): batch size
        test_ratio ([float]): validation set ratio

    Returns:
        Tuple[DataLoader, DataLoader]: train and validation dataloader
    """

    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size = int(train_ratio*len(dataset))
    valid_size = len(dataset)-train_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size[0], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size[1], shuffle=True)

    return train_dataloader, valid_dataloader

def build_test_dataloaders(
    input_ids: Tensor,
    attention_masks: Tensor, 
    labels: Tensor
) -> DataLoader:
    """
    This function takes in tokenized copus, and returns train and validation dataloaders.

    Args:
        input_ids[Tensor]: input ids
        attention_masks[Tensor]: attention masks

    Returns:
        DataLoader: test dataloader
    """

    dataset = TensorDataset(input_ids, attention_masks, labels)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    return test_dataloader

def dummy_data_collector(features):
    batch = {}
    batch['input_ids'] = torch.stack([f[0] for f in features])
    batch['attention_mask'] = torch.stack([f[1] for f in features])
    batch['labels'] = torch.stack([f[2] for f in features])
    
    return batch

def _split_sequence(text, max_len):
    words = text.split()
    lines = []
    current = []
    for word in words:
        if len(current) >= max_len:
            lines.append(' '.join(current))
            current = [word]
        else:
            current.append(word+' ')
    if len(current) >= 300:
        lines.append(' '.join(current))
    return lines

def split_training_data(df, max_len):
    split_texts_tuples = []
    for row in df.iterrows():
        split_texts = _split_sequence(row[1]['text'], max_len)
        for text in split_texts:
            split_texts_tuples.append((text, row[1]['class_id']))
    df = pd.DataFrame(split_texts_tuples, columns=['text', 'label'])
    return df
