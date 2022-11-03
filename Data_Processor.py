from datasets import load_dataset, Dataset
from torch import nn
from transformers import RobertaTokenizer, PreTrainedModel, RobertaForSequenceClassification, RobertaModel
import torch
from torch.utils.data import DataLoader


def res_indices(conflict):
    code_lines = conflict['ours'][:] + conflict['theirs'][:]
    indices = []
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
    pad_lines = example['lines']
    pad_label = example['label']
    valid_len = len(example['label'])
    for i in range(max_len - len(example['lines'])):
        pad_lines = pad_lines + ['<pad>']
    for i in range(max_len - valid_len):
        pad_label = pad_label + [0]
    return {
        'lines': pad_lines,
        'label': pad_label,
        'valid_len': valid_len
    }


def tokenize_func(example):
    return tokenizer(example['lines'], padding='max_length', truncation=True, return_tensors="pt")


def embedding_func(example):
    return {'hidden_state': codeBert(torch.tensor(example['input_ids']), torch.tensor(example['attention_mask']))['pooler_output']}


dataset_path = 'G:/merge/dataset/tmp1'
bertPath = 'G:/merge/model/CodeBERTa-small-v1'

dataset = load_dataset("json", data_files="G:/study/deeplearning/d2l-pytorch/data/1.json", field="mergeTuples")
tokenizer = RobertaTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
codeBert = RobertaModel.from_pretrained(bertPath)
max_len = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


label_dataset = dataset.map(res_indices, remove_columns=['ours', 'theirs', 'resolve', 'path', 'base'])
# max_len = get_max_len(label_dataset['train'])
filter_dataset = label_dataset.filter(lambda data: len(data['lines']) < max_len and len(data['label']) < max_len)
padding_dataset = filter_dataset.map(lambda x: padding(x, max_len))
tokenized_dataset = padding_dataset.map(tokenize_func)

for param in codeBert.parameters():
    param.requires_grad_(False)


dataset_size = 1

ds_cpu = tokenized_dataset['train'].select(range(dataset_size)).map(embedding_func, remove_columns=['lines', 'input_ids', 'attention_mask'])
ds_cpu = ds_cpu.map(lambda e: {'hidden_state': e['hidden_state'][0]})
print(len(ds_cpu['hidden_state']))
print(len(ds_cpu['hidden_state'][0]))
ds_cpu.save_to_disk(dataset_path)


# codeBert.to(device)
# ds_gpu = tokenized_dataset['train'].with_format("torch", device=device)
# emb_gpu = ds_gpu.select(range(dataset_size)).map(embedding_func, remove_columns=['lines', 'input_ids', 'attention_mask'])
# emb_cpu = emb_gpu.with_format("torch", device=torch.device('cpu'))
# print(emb_cpu)
# print(emb_cpu[0]['label'].keys())
# emb_cpu.save_to_disk("G:/study/deeplearning/d2l-pytorch/data")
