import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from prep_bert import BertEncoder, build_dataloaders
from fine_tune_bert import fine_tune_bert
import pickle5 as pickle

if __name__ == '__main__':
    with open('data.pickle', 'rb') as handle:
        df = pickle.load(handle)
    texts = df['text'].tolist()
    label = df['class_id'].apply(lambda x: int(x))
    texts, _, label, _ = train_test_split(texts, label, test_size=0.2, stratify=label, random_state=2020)

    dataset = BertEncoder(
        tokenizer=RobertaTokenizer.from_pretrained(
            'roberta-base', 
            do_lower_case=False), 
        input_data=texts
    )

    data = dataset.tokenize(max_len=510)
    input_ids, attention_masks = data
    label = torch.Tensor(label.to_list()).long()
    train_dataloader, val_dataloader = build_dataloaders(
        input_ids=input_ids,
        attention_masks=attention_masks,
        labels=label,
        batch_size=(16, 4),
        train_ratio=0.8
    )

    bert_model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        output_attentions=False,
        output_hidden_states=False,
        num_labels = 4
    )

    optimizer = AdamW(bert_model.parameters(), lr=2e-5, eps=1e-8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trained_model, stats = fine_tune_bert(
        train_dataloader=train_dataloader, 
        valid_dataloader=val_dataloader, 
        model=bert_model,
        optimizer=optimizer,
        save_model_path='model/trained_roberta_model.pt',
        save_stats_dict_path='logs/roberta_statistics.json',
        device = device
    )