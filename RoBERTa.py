import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from prep_bert import *
from fine_tune_bert_sc import *
import pickle5 as pickle
import pandas as pd
from torch.utils.data import TensorDataset

if __name__ == '__main__':
    with open('data.pickle', 'rb') as handle:
        df = pickle.load(handle)
    texts = df['text'].tolist()
    labels = df['class_id'].apply(lambda x: int(x))
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=2020)

    train_dataset = BertEncoder(
        tokenizer=RobertaTokenizer.from_pretrained('roberta-base'), 
        input_data=train_texts
    )
    test_dataset = BertEncoder(
        tokenizer=RobertaTokenizer.from_pretrained('roberta-base'), 
        input_data=test_texts
    )

    train_data = train_dataset.tokenize(max_len=510)
    test_data = test_dataset.tokenize(max_len=510)

    train_input_ids, train_attention_masks = train_data
    test_input_ids, test_attention_masks = test_data

    train_labels = torch.Tensor(train_labels.to_list()).long()
    test_labels = torch.Tensor(test_labels.to_list()).long()

    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        output_attentions=False,
        output_hidden_states=False,
        num_labels = 4
    )

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=8,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=dummy_data_collector
    )

    trainer.train()
    trainer.save_model('./model')
    trainer.evaluate()
    