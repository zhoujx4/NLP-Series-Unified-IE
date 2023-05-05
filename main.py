"""
@Time : 2023/4/2411:11
@Auth : zhoujx
@File ：main.py
@DESCRIPTION:

"""
import json
import os

import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pickle
from random import shuffle

from torch.optim import AdamW
from tqdm import trange

from dataloader.UsmDataloader import UsmDataloader
from models.UsmModel import UsmModel
from utils.argparse import get_argparse
from utils.utils import set_seeds, RunningAverage, RunningEMA
from loguru import logger


def load_train_data(bl, args, dtype='train'):
    if not os.path.isfile(os.path.join(args.data_dir, 'train.pkl')) or True:
        train_data = bl.load_data(dtype)
        ner_train_data, ent_snt_train_data, re_train_data, triplet_train_data = bl.tokenize_train_data(train_data)
        with open(os.path.join(args.data_dir, 'train.pkl'), 'wb') as f:
            train_pkl = {'ner': ner_train_data,
                         'ent_snt': ent_snt_train_data,
                         're': re_train_data,
                         'triplet': triplet_train_data}
            pickle.dump(train_pkl, f)
    else:
        with open(os.path.join(args.data_dir, 'train.pkl'), 'rb') as f:
            train_pkl = pickle.load(f)
            ner_train_data, ent_snt_train_data, re_train_data, triplet_train_data = train_pkl['ner'], \
                                                                                    train_pkl['ent_snt'], \
                                                                                    train_pkl['re'], \
                                                                                    train_pkl['triplet']
    return ner_train_data, ent_snt_train_data, re_train_data, triplet_train_data


def eval(model, bl, data_type, args, epoch, device, batch_size=128):
    pred_data = extract(model, bl, data_type, args, batch_size, device)
    with open(os.path.join(args.model_dir, 'epoch_{}'.format(epoch)), 'w', encoding='utf-8') as f:
        num_pred, num_recall, num_correct = 1e-10, 1e-10, 1e-10
        for sample in pred_data:
            gold_spos = sorted(sample['spos'])
            pred_spos = sorted(sample['pred_spos'])
            a, b, c = compute_corrects(pred_spos, gold_spos)
            if not only_bad_cases or not (a == b == c):
                f.write(sample['text'] + '\n')
                f.write(str(gold_spos) + '\n')
                f.write(str(pred_spos) + '\n\n')
            num_pred += a
            num_recall += b
            num_correct += c
        precision, recall, f1 = compute_metrics(num_pred, num_recall, num_correct)
        logging.info('{}: Precision: {:5.4f}, Recall: {:5.4f}, F1: {:5.4f}'.format(data_type, precision, recall, f1))
    return f1


def extract(model, bl, data_type, args, batch_size, device):
    model.eval()
    '''
    data: List(dict: 内容|类型|答案)
    eval_data: List(dict: token|input_ids|token_type_ids|attention_mask|labels|entity_type_indexes|id)
    '''
    with torch.no_grad():
        data = bl.load_data(data_type)
        true_

        for sample in data:
            sample['预测结果'] = []
        # subs extraction
        eval_data = bl.tokenize_train_data(data)
        eval_data = [x2 for x1 in eval_data for x2 in x1]
        eval_bl = bl.eval_batch_loader(eval_data, args.batch_size)  # 出来后是一个列表
        preds = []
        for i, batch_data in enumerate(eval_bl):
            batch_eval_data = [tmp for tmp in eval_data[i * batch_size: (i + 1) * batch_size]]
            batch_data = tuple(tmp.to(device) for tmp in batch_data)
            outputs_gt_0 = model.get_res(batch_data)  # 得到预测矩阵为 1 的位置
            preds += decode(batch_eval_data, outputs_gt_0)
    return data


def main(args):
    print(json.dumps(vars(args), indent=4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bl = UsmDataloader(args)

    # Model
    model = UsmModel(pretrain_model_path=args.model_name_or_path,
                     max_length=args.max_length,
                     schema=bl.schema,
                     with_adversarial_training=True,
                     matrix_hidden_size=400,
                     head_size=64,
                     dropout_rate=0.15)
    model.encoder.resize_token_embeddings(len(bl.tokenizer))
    model.to(device)

    train_dtype = 'train'
    valid_dtype = 'valid'
    test_dtype = 'test'

    if args.do_train_and_eval:
        ## Train data
        ner_train_data, ent_snt_train_data, re_train_data, triplet_train_data = load_train_data(bl,
                                                                                                args,
                                                                                                train_dtype)
        train_dataloaders = bl.batch_loader(ner_train_data, ent_snt_train_data, re_train_data, triplet_train_data,
                                            args.batch_size)  # 出来后是一个列表

        num_batchs_per_task = [len(train_dataloader) for train_dataloader in train_dataloaders]
        logger.info('num of batch per task for train: {}'.format(num_batchs_per_task))
        train_task_ids = sum([[i] * num_batchs_per_task[i] for i in range(len(train_dataloaders))], [])
        # Optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'names': [n for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'names': [n for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        args.epoch_steps = sum(num_batchs_per_task)
        args.total_steps = args.epoch_steps * args.epoch
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate)

        for epoch in range(args.epoch):
            ## Train
            model.train()
            t = trange(args.epoch_steps, desc='Epoch {} -Train'.format(epoch))
            avg_loss = RunningAverage()
            train_iters = [iter(tmp) for tmp in train_dataloaders]  # to use next and reset the iterator
            shuffle(train_task_ids)
            scales = [1, 1, 1, 1]
            tasks_avg_loss = [RunningEMA() for _ in range(len(train_dataloaders))]
            for step in t:
                if epoch == 0 and step < 100:
                    task_id = train_task_ids[step]
                else:
                    if args.do_sampling:
                        task_id = np.random.choice([0, 1, 2], p=[tmp / 3 for tmp in scales])
                    else:
                        task_id = train_task_ids[step]
                try:
                    batch_data = next(train_iters[task_id])
                except StopIteration:
                    train_iters[task_id] = iter(train_dataloaders[task_id])
                    batch_data = next(train_iters[task_id])
                new_tmp = []
                batch_data = tuple(tmp.to(device) for tmp in batch_data)
                loss = model(batch_data, task_id)
                avg_loss.update(loss.item())
                tasks_avg_loss[task_id].update(loss.item())
                if args.mtl_strategy == 'norm':
                    scales = [1, 1, 1, 1]
                elif args.mtl_strategy == 'avg_sum_loss_lr':
                    scales = [a() / b for a, b in zip(model.avg_metrics, num_batchs_per_task)]
                    scales = [4 / sum(scales) * tmp for tmp in scales]
                    scales = [tmp / scales[3] for tmp in scales]
                else:
                    raise Exception('wrong mtl strategy')
                if not args.do_sampling:
                    loss = scales[task_id] * loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                t.set_postfix(loss='{:5.4f}'.format(loss.item()), \
                              avg_loss='{:5.4f}'.format(avg_loss()), \
                              scales='{:5.2f}, {:5.2f}, {:5.2f}'.format(*[tmp for tmp in scales]), \
                              tasks_avg_loss='{:5.4f}, {:5.4f}, {:5.4f}'.format(*[tmp() for tmp in tasks_avg_loss]))
            f1 = eval(model, bl, valid_dtype, args, epoch, device, 128)


if __name__ == '__main__':
    args = get_argparse().parse_args()
    set_seeds(4)
    main(args)
