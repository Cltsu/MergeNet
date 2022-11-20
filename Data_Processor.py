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
    code_lines.append('<sep>')
    if not conflict['theirs'] is None:
        code_lines += conflict['theirs'][:]
    indices = []
    if not conflict['resolve'] is None:
        for line in conflict['resolve']:
            for i in range(len(code_lines)):
                if line == code_lines[i]:
                    indices.append(i)
                    break
    return {'label': indices}


def get_max_len(dataset):
    dataloader = DataLoader(dataset, batch_size=1)
    max_len1 = 0
    for data in dataloader:
        max_len1 = max(max_len1, len(data['label']), len(data['lines']))
    return max_len1


def padding(example, max_len):
    valid_len = len(example['label'])
    pad_lines = example['lines'] + ['<eoc>']
    pad_label = example['label'] + [len(example['lines'])]
    pad_editseq = example['editseq'] + ['<eoc>']
    for i in range(max_len - len(example['lines']) - 1):
        pad_lines = pad_lines + ['<pad>']
        pad_editseq = pad_editseq + ['<pad>']
    for i in range(max_len - valid_len - 1):
        pad_label = pad_label + [0]

    # padding resolve
    pad_resolve = example['resolve']
    for i in range(max_len - len(example['resolve'])):
        pad_resolve = pad_resolve + ['<pad>']

    return {
        'resolve': pad_resolve,
        'lines': pad_lines,
        'label': pad_label,
        'valid_len': valid_len + 1
    }


# def add_edit_token(example):
#     def edit_info(ours, theirs, base):
#         ret_ours = []
#         for line in ours:
#             tmp = line
#             if line in base and line in theirs:
#                 tmp = '<==>' + tmp
#             if line in base and line not in theirs:
#                 tmp = '<=>' + tmp
#             if line not in base and line in theirs:
#                 tmp = '<+=>' + tmp
#             if line not in base and line not in theirs:
#                 tmp = '<+>' + tmp
#             ret_ours.append(tmp)
#         return ret_ours
#
#     return {
#         'ours': edit_info(example['ours'], example['theirs'], example['base']),
#         'theirs': edit_info(example['theirs'], example['ours'], example['base']),
#     }


def edit_seq(example):
    def edit_info(ours, theirs, base):
        editseq = []
        for line in ours:
            if line in base and line in theirs:
                editseq.append('<==>')
            if line in base and line not in theirs:
                editseq.append('<=>')
            if line not in base and line in theirs:
                editseq.append('<+=>')
            if line not in base and line not in theirs:
                editseq.append('<+>')
        return editseq

    return {
        'editseq_ours': edit_info(example['ours'], example['theirs'], example['base']),
        'editseq_theirs': edit_info(example['theirs'], example['ours'], example['base']),
    }


def tokenize_func(example):
    return tokenizer(example['lines'], padding='max_length', truncation=True, return_tensors="pt")


def tokenize_editseq(example):
    tokenized = tokenizer(example['lines'], padding='max_length', truncation=True, return_tensors="pt")
    return {
        'editseq_input_ids': tokenized['input_ids'],
        'editseq_attention_mask': tokenized['attention_mask']
    }

# def embedding_func(example):
#     return {'hidden_state': codeBert(torch.tensor(example['input_ids']), torch.tensor(example['attention_mask']))[
#         'pooler_output']}


dataset_path = 'G:/merge/dataset/processed_add_special'
bertPath = 'G:/merge/model/CodeBERTa-small-v1'
load_path = "G:/merge/dataset/raw_data/interleaving.json"
max_len = 30

dataset = load_dataset("json", data_files=load_path, field="mergeTuples")
special_tokens_dict = {'additional_special_tokens': ['<sep>', '<eoc>', '<==>', '<=>', '<+=>', '<+>']}
tokenizer = RobertaTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
tokenizer.add_special_tokens(special_tokens_dict)

# codeBert = RobertaModel.from_pretrained(bertPath)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('compute resolution indices')
# label_dataset = dataset.map(res_indices, remove_columns=['ours', 'theirs', 'resolve', 'path', 'base'])
label_dataset = dataset.map(res_indices)
print(len(label_dataset['train']))

print('add edit info for each code line')
edit_dataset = label_dataset.map(edit_seq, remove_columns=['path', 'base'])
edit_dataset = edit_dataset.map(lambda data: {
    'lines': data['ours'] + ['<sep>'] + data['theirs'],
    'editseq': data['editseq_ours'] + ['<sep>'] + data['editseq_theirs']
}, remove_columns=['ours', 'theirs', 'editseq_ours', 'editseq_theirs'])

print('filter large conflicts')
filter_dataset = edit_dataset.filter(
    lambda data: len(data['lines']) < max_len - 1 and len(data['label']) < max_len - 1)  # max_len - 1 因为要给stop token留位置
print(len(filter_dataset['train']))

print('padding')
padding_dataset = filter_dataset.map(lambda x: padding(x, max_len))
print(len(padding_dataset['train']))

print("tokenizing")
# tokenized_dataset = padding_dataset.map(tokenize_func, remove_columns=['lines'])
tokenized_dataset = padding_dataset.map(tokenize_func)
tokenized_dataset = tokenized_dataset.map(tokenize_editseq, remove_columns=['editseq'])
print(len(tokenized_dataset['train']))

# print(tokenizer.decode(tokenized_dataset['train'][1]['input_ids'][0]))
for key in tokenized_dataset['train'][1].keys():
    print(key + str(tokenized_dataset['train'][1][key]))

tokenized_dataset.save_to_disk(dataset_path)

# print(len(tokenized_dataset['train'][0]['input_ids']))
# print(len(tokenized_dataset['train'][0]['input_ids'][0]))
# dataloader = DataLoader(tokenized_dataset['train'], batch_size=1)
# for data in dataloader:
#     print(data)
#     for i in data['input_ids']:
#         print(len(i))


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
