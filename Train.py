"""

Pytorch implementation of Pointer Network.

http://arxiv.org/pdf/1506.03134v1.pdf.

"""

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets import Dataset, load_from_disk
from transformers import RobertaTokenizer, PreTrainedModel, RobertaForSequenceClassification, RobertaModel
import numpy as np
import argparse
from tqdm import tqdm

from PointerNet import PointerNet

parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

# Data
# parser.add_argument('--train_size', default=100000, type=int, help='Training data size')
# parser.add_argument('--val_size', default=10000, type=int, help='Validation data size')
# parser.add_argument('--test_size', default=10000, type=int, help='Test data size')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
# Train
parser.add_argument('--nof_epoch', default=10, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
# GPU
parser.add_argument('--gpu', default=True, action='store_true', help='Enable gpu')
# TSP
# parser.add_argument('--nof_points', type=int, default=5, help='Number of points in TSP')
# Network
parser.add_argument('--embedding_size', type=int, default=768, help='Embedding size')
parser.add_argument('--hiddens', type=int, default=1024, help='Number of hidden units')
parser.add_argument('--nof_lstms', type=int, default=2, help='Number of LSTM layers')
parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
parser.add_argument('--bidir', default=False, action='store_true', help='Bidirectional')
parser.add_argument('--embed_batch', type=int, default=4)
parser.add_argument('--bert_grad', type=bool, default=False)
parser.add_argument('--codeBERT', type=str, default='codeBERT/CodeBERTa-small-v1', help='path of codeBERT')
parser.add_argument('--dataset', type=str, default='data', help='path of dataset')

params = parser.parse_args()

dataset_path = params.dataset
codeBERT_path = params.codeBERT

if params.gpu and torch.cuda.is_available():
    USE_CUDA = True
    print('Using GPU, %i devices.' % torch.cuda.device_count())
else:
    USE_CUDA = False

embed_model = RobertaModel.from_pretrained(codeBERT_path)

if not params.bert_grad:
    for param in embed_model.parameters():
        param.requires_grad_(False)

model = PointerNet(params.embedding_size,
                   params.hiddens,
                   params.nof_lstms,
                   params.dropout,
                   params.embed_batch,
                   embed_model,
                   params.bidir, )

dataset = load_from_disk(dataset_path).with_format(type='torch')
# dataset = dataset.select(range(30))
# dataset.train_test_split(test_size=0.1)


dataloader = DataLoader(dataset,
                        batch_size=params.batch_size,
                        shuffle=True,
                        num_workers=0)

if USE_CUDA:
    model.cuda()
    net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

CCE = torch.nn.CrossEntropyLoss()
model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()),
                         lr=params.lr)

losses = []
max_len = 30


def max_index(tl):
    ret = 0
    max_prob = tl[0]
    for ii in range(1, len(tl)):
        if tl[ii] > max_prob:
            max_prob = tl[ii]
            ret = ii
    return ret


for epoch in range(params.nof_epoch):
    batch_loss = []
    batch_acc = 0
    iterator = tqdm(dataloader, unit='Batch')

    for i_batch, sample_batched in enumerate(iterator):

        iterator.set_description('Epoch %i/%i' % (epoch + 1, params.nof_epoch))

        input_batch = Variable(sample_batched['input_ids'])
        att_batch = Variable(sample_batched['attention_mask'])
        target_batch = Variable(sample_batched['label'])

        if USE_CUDA:
            input_batch = input_batch.cuda()
            att_batch = att_batch.cuda()
            target_batch = target_batch.cuda()

        o, p = model([input_batch, att_batch])

        valid_len_batch = sample_batched['valid_len']
        cur_batch_size = len(valid_len_batch)

        pred = torch.tensor([max_index(probs) for example in o for probs in example], dtype=torch.int64)
        pred = pred.view(cur_batch_size, max_len)
        if USE_CUDA:
            pred = pred.cuda()
        for i in range(len(valid_len_batch)):
            if pred[i][0:valid_len_batch[i]].equal(target_batch[i][0:valid_len_batch[i]]):
                batch_acc += 1

        mask_tensor = torch.zeros(size=(o.size()[:2]))
        for i in range(o.size()[0]):
            mask_tensor[i][0:valid_len_batch[i]] = 1

        if USE_CUDA:
            mask_tensor = mask_tensor.cuda()

        mask_tensor = mask_tensor.view(-1)

        o = o.contiguous().view(-1, o.size()[-1])
        target_batch = target_batch.view(-1)

        loss = CCE(o, target_batch)

        loss = torch.mul(loss, mask_tensor)
        loss = loss.sum() / valid_len_batch.sum()

        losses.append(loss.data)
        batch_loss.append(loss.data.cpu())

        model_optim.zero_grad()
        loss.backward()
        model_optim.step()

        iterator.set_postfix(loss='{}'.format(loss.data))

    print('Epoch {0} / {1}, average loss : {2} , average accuracy : {3}'.
          format(epoch + 1, params.nof_epoch, np.average(batch_loss), batch_acc / len(dataset)))
