from ast import literal_eval

from datasets import load_dataset, Dataset
from torch import nn
from transformers import RobertaTokenizer, PreTrainedModel, RobertaForSequenceClassification, RobertaModel
import torch
from torch.utils.data import DataLoader


def res_indices(conflict):
    code_lines = []
    if not conflict['ours'] is None:
        code_lines = conflict['ours'][:]
    if not conflict['theirs'] is None:
        code_lines += conflict['theirs'][:]
    indices = []
    if not conflict['resolve'] is None:
        for line in conflict['resolve']:
            for i in range(len(code_lines)):
                if line == code_lines[i]:
                    indices.append(i)
                    break
    return {'label': indices, 'lines': code_lines}


def get_max_len(dataset):
    dataloader = DataLoader(dataset, batch_size=1)
    max_len1 = 0
    for data in dataloader:
        max_len1 = max(max_len1, len(data['label']), len(data['lines']))
    return max_len1


def padding(example, max_len):
    valid_len = len(example['label'])
    pad_lines = example['lines'] + ['<eos>']
    pad_label = example['label'] + [valid_len]
    for i in range(max_len - len(example['lines']) - 1):
        pad_lines = pad_lines + ['<pad>']
    for i in range(max_len - valid_len - 1):
        pad_label = pad_label + [0]
    return {
        'lines': pad_lines,
        'label': pad_label,
        'valid_len': valid_len + 1
    }


def tokenize_func(example):
    return tokenizer(example['lines'], padding='max_length', truncation=True, return_tensors="pt")


def embedding_func(example):
    return {'hidden_state': codeBert(torch.tensor(example['input_ids']), torch.tensor(example['attention_mask']))[
        'pooler_output']}


dataset_path = 'G:/merge/dataset/noEmbed'
bertPath = 'G:/merge/model/CodeBERTa-small-v1'
load_path = "G:/merge/dataset/raw_data/data.json"
max_len = 30

dataset = load_dataset("json", data_files=load_path, field="mergeTuples")
# dataset = load_dataset('csv', data_files=load_path)
tokenizer = RobertaTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
codeBert = RobertaModel.from_pretrained(bertPath)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('compute resolution indices')
label_dataset = dataset.map(res_indices, remove_columns=['ours', 'theirs', 'resolve', 'path', 'base'])
print(len(label_dataset['train']))

print('filter large conflicts')
filter_dataset = label_dataset.filter(lambda data: len(data['lines']) < max_len - 1 and len(data['label']) < max_len - 1) # max_len - 1 因为要给stop token留位置
print(len(filter_dataset['train']))

print('padding')
padding_dataset = filter_dataset.map(lambda x: padding(x, max_len))
print(len(padding_dataset['train']))

print("tokenizing")
tokenized_dataset = padding_dataset.map(tokenize_func, remove_columns=['lines'])
print(len(tokenized_dataset['train']))


# print(len(tokenized_dataset['train'][0]['input_ids']))
# print(len(tokenized_dataset['train'][0]['input_ids'][0]))
# dataloader = DataLoader(tokenized_dataset['train'], batch_size=1)
# for data in dataloader:
#     print(data)
#     for i in data['input_ids']:
#         print(len(i))
tokenized_dataset.save_to_disk(dataset_path)

# for param in codeBert.parameters():
#     param.requires_grad_(False)
#
# if torch.cuda.is_available():
#     USE_CUDA = True
# else:
#     USE_CUDA = False
#
# print('---------------embedding begin--------------------')
# if USE_CUDA:
#     codeBert.to(device)
#     ds_gpu = tokenized_dataset['train'].with_format("torch", device=device)
#     emb_gpu = ds_gpu.map(embedding_func,
#                          remove_columns=['lines', 'input_ids', 'attention_mask'])
#     Dataset.from_dict(emb_gpu.to_dict()).save_to_disk(dataset_path)
# else:
#     ds_cpu = tokenized_dataset['train'].map(embedding_func,
#                                             remove_columns=['lines', 'input_ids',
#                                                             'attention_mask'])
#     ds_cpu = ds_cpu.map(lambda e: {'hidden_state': e['hidden_state'][0]})
#     ds_cpu.save_to_disk(dataset_path)
