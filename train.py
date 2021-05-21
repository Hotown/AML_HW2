import os
import random

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import (AdamW, BertForSequenceClassification, BertTokenizer,
                          get_linear_schedule_with_warmup)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda')

def encode_fn(text_list, tokenizer):
    all_input_ids = []
    for text in text_list:
        input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=180, pad_to_max_length=True, return_tensors='pt')
        all_input_ids.append(input_ids)
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(pred_flat, labels_flat)

def flat_f1_score(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(pred_flat, labels_flat, average='macro')

for dir_name, _, file_names in os.walk('/root/hotown/hw2_v2/data'):
    # df_train = pd.read_csv(os.path.join(dir_name, 'train.txt'))
    # df_test = pd.read_csv(os.path.join(dir_name, 'test.txt'))

    # df_train['final_text'] = df_train['review_text']
    # df_test['final_text'] = df_test['review_text']

    df_train = pd.read_csv(os.path.join(dir_name, 'cleaned_train.csv'))
    df_test = pd.read_csv(os.path.join(dir_name, 'cleaned_test.csv'))

    df_train['final_text'] = df_train['text_clean']
    df_test['final_text'] = df_test['text_clean']

    text_values = df_train['final_text'].values
    text_labels = df_train['fit'].values
    text_labels[np.where(text_labels == 'fit')] = 0
    text_labels[np.where(text_labels == 'small')] = 1
    text_labels[np.where(text_labels == 'large')] = 2
    labels = np.stack(text_labels).astype(np.long)

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenizer = BertTokenizer.from_pretrained('./weights/-3', do_lower_case=True)

    print(f'Original Text: {text_values[1]}')
    print(f'Tokenized Text: {tokenizer.tokenize(text_values[1])}')
    print(f'Token IDs: {tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_values[1]))}')

    print(f'Adding Special Tokens Using Encode Func: {tokenizer.encode(text_values[1])}')

    df_train['length'] = df_train['final_text'].str.len()
    df_test['length'] = df_test['final_text'].str.len()

    sns.set_style('whitegrid', {'axes.grid' : False})
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(18, 6)

    sns.distplot(df_train['length'], color='#20A387',ax=ax[0])
    sns.distplot(df_test['length'], color='#440154',ax=ax[1])

    fig.suptitle('Length of Tweets', fontsize=14)
    ax[0].set_title('Train')
    ax[1].set_title('Test')

    # plt.savefig('data_length')
    not_null_index = np.where(pd.isnull(text_values) == False)[0]
    text_values = text_values[not_null_index]
    labels = labels[not_null_index]

    epochs = 4
    batch_size = 32

    # Test
    # text_values = text_values[:1024]
    # labels = labels[:1024]

    all_input_ids = encode_fn(text_values, tokenizer)
    labels = torch.tensor(labels)

    dataset = TensorDataset(all_input_ids, labels)
    train_size = int(0.90 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', problem_type = "single_label_classification", num_labels=3, output_attentions=False, output_hidden_states=False)
    model = BertForSequenceClassification.from_pretrained('./weights/-3', problem_type = "single_label_classification", num_labels=3, output_attentions=False, output_hidden_states=False)
    model.cuda()

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        model.train()
        total_loss, total_val_loss = 0, 0
        total_eval_acc = 0
        total_eval_f1 = 0
        for step, batch in enumerate(train_dataloader):
            model.zero_grad()
            outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0]>0).to(device), labels=batch[1].to(device))
            loss = outputs.loss
            logits = outputs.logits
            print(f'Train loss     : {loss}')
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()

        for i, batch in enumerate(val_dataloader):
            with torch.no_grad():
                outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0]>0).to(device), labels = batch[1].to(device))
                loss = outputs.loss
                logits = outputs.logits
                logits = logits.detach().cpu().numpy()
                label_ids = batch[1].cpu().numpy()
                total_val_loss += loss
                total_eval_acc += flat_accuracy(logits, label_ids)
                total_eval_f1 += flat_f1_score(logits, label_ids)

        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_accuracy = total_eval_acc / len(val_dataloader)
        avg_val_f1 = total_eval_f1 / len(val_dataloader)
        
        print(f'Train loss     : {avg_train_loss}')
        print(f'Validation loss: {avg_val_loss}')
        print(f'Accuracy: {avg_val_accuracy:.2f}')
        print(f'F1 Scoure: {avg_val_f1:.2f}')
        print('\n')
        model.save_pretrained('/root/hotown/hw2_v2/weights/' + '-' + str(epoch)+ '-v2')
        tokenizer.save_pretrained('/root/hotown/hw2_v2/weights/' + '-' + str(epoch)+ '-v2')
