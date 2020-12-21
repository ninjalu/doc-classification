from transformers import get_linear_schedule_with_warmup
from typing import Optional, Tuple, Union, Dict
from torch.utils.data import DataLoader
import json
import torch
from tqdm import tqdm

def fine_tune_bert(
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 5,
    lr_scheduler_warmup_steps: int = 0,
    save_model_path: str = None,
    save_stats_dict_path: str = None,
    device: str = None
) -> Tuple[torch.nn.Module, Dict[str, Dict[str, Union[float, str]]]]:
    """
    This function performs fine tuning of BERT for masked language modelling. It returns the trained model as well as a dictionary with evaluation statistics
    at each opochs.

    Args:
        train_dataloader (DataLoader): dataloader containing input ids, token type ids, attention masks.
        valid_dataloader (DataLoader): dataloader containing input ids, token type ids, attention masks.
        model (torch.nn.Module): 
        optimizer (torch.optim.Optimizer): 
        train_ratio (float, optional): Defaults to 0.8.
        epochs (int, optional): Defaults to 4.
        lr_scheduler_warmup_steps (int, optional): step at which learning rate scheduler kicks in. Defaults to 0.
        save_model_path (str, optional): Defaults to None.
        save_stats_dict_path (str, optional): Defaults to None.
        device (str, optional): Defaults to None.

    Returns:
        Tuple[torch.nn.Module, Dict[str, Dict[str, Union[float, str]]]]: trained model and training stats
    """
    training_steps = epochs * len(train_dataloader) # epoch * num of batches
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = lr_scheduler_warmup_steps,
        num_training_steps = training_steps
    )
    model = model.to(device)
    training_stats = {}
    model.train()
    for epoch in tqdm(range(epochs)):
        cumulative_train_loss_per_epoch = 0
        for batch in train_dataloader:
            input_ids, attention_masks = batch
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            model.zero_grad()
            output = model(
                input_ids=input_ids,
                token_type_ids=None,
                attention_mask=attention_masks,
                labels=input_ids
            )
            loss = output[0]
            cumulative_train_loss_per_epoch += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            lr_scheduler.step()
        
        average_training_loss_per_batch = cumulative_train_loss_per_epoch / len(train_dataloader)
        
        pred_ids = torch.tensor([], dtype=torch.long, device=device)  # initialising tensors for storing results
        true_ids = torch.tensor([], dtype=torch.long, device=device)  # initialising tensors for storing results
        
        cumulative_eval_loss_per_epoch = 0
        
        for batch in valid_dataloader:
            input_ids, attention_masks = batch
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            with torch.no_grad():
                output = model(
                    input_ids=input_ids,
                    token_type_ids=None,
                    attention_mask=attention_masks,
                    labels=input_ids
                )
                loss = output[0]
                logits = output[1]
                cumulative_eval_loss_per_epoch += loss.item()

                pred_id = torch.argmax(logits, dim=1)
                pred_ids = torch.cat((pred_ids, pred_id))
                true_ids = torch.cat((true_ids, input_ids))
        
        average_validation_accuracy_per_epoch = int(sum(pred_ids==true_ids))/len(valid_dataloader)
        average_val_loss_per_batch = cumulative_eval_loss_per_epoch/len(valid_dataloader)

        training_stats[f"epoch_{epoch + 1}"] = {
            "training_loss": average_training_loss_per_batch,
            "valid_loss": average_val_loss_per_batch,
            "valid_accuracy": average_validation_accuracy_per_epoch
        }

        if save_model_path is not None:
            torch.save(model.state_dict(), save_model_path)
        if save_stats_dict_path is not None:
            with open(save_stats_dict_path, 'w') as file:
                json.dump(training_stats, file)

        return model, training_stats
