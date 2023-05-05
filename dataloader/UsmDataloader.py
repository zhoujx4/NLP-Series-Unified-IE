"""
@Time : 2023/4/2411:15
@Auth : zhoujx
@File ：UsmDataloader.py
@DESCRIPTION:

"""
import itertools
import json
import os

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer


def pad_collate_fn(batch):
    # batch_size == len(batch)
    batch = [torch.stack(x) for x in list(zip(*batch))]
    input_ids = batch[0]
    token_type_ids = batch[1]
    attention_mask = batch[2]
    if len(batch) == 3:
        return [input_ids, token_type_ids, attention_mask]
    elif len(batch) == 4:
        matrixs = batch[3]
        return [input_ids, token_type_ids, attention_mask, matrixs]
    else:
        raise Exception('wrong length of batch')

    # if len(batch) == 4:
    #     heads = batch[2][:, :length]
    #     tails = batch[3][:, :length]
    #     return [token_ids, token_types, heads, tails]
    # elif len(batch) == 3:
    #     label = batch[2]
    #     return [token_ids, token_types, label]
    # elif len(batch) == 2:
    #     return [token_ids, token_types]
    # else:
    #     raise Exception('wrong length of batch')


class UsmDataloader(object):
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.max_length = args.max_length
        self.tokenizer = self.load_tokenizer(args.model_name_or_path)
        self.schema = self.load_schema()
        self.ner_index = [int(x) for x in self.schema['index_2_ner'].keys()]
        self.triplet_index = [int(x) for x in self.schema['index_2_triplet'].keys()]
        self.tri_snt_index = [int(x) for x in self.schema['index_2_tri_snt'].keys()]
        self.sentiment_index = [int(x) for x in self.schema['index_2_sentiment'].keys()]
        self.schema_tokens_length = self.schema['schema_tokens_length']

    def load_tokenizer(self, model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=False)
        tokenizer.add_tokens(['[industry]', '[ner]', '[triplet]', '[tri_snt]', '[sentiment]'])

        return tokenizer

    def load_data(self, dtype, n=None):
        file_path = os.path.join(self.data_dir, dtype + '.csv')
        df = pd.read_csv(file_path, nrows=n)
        df = df.sample(200)
        data = df.to_dict(orient='records')
        data = self._filter_unvalid_data(data)
        data = self._filter_data(data)
        return data

    def load_schema(self):
        with open(os.path.join(self.data_dir, 'schema.json'), 'r', encoding='utf-8') as f:
            schema = json.load(f)
        return schema

    def tokenize_train_data(self, data):
        ner_data = []
        ent_snt_data = []
        re_data = []
        triplet_data = []
        for idx, sample in tqdm(enumerate(data), desc='Tokenizing train data...'):
            # ['内容', '类型', '答案']
            # encodings = ['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'overflow_to_sample_mapping']
            encodings = self.tokenizer(self.schema['schema'],
                                       sample['内容'],
                                       add_special_tokens=True,
                                       is_split_into_words=False,
                                       max_length=self.max_length,
                                       truncation='only_second',
                                       return_overflowing_tokens=True,
                                       return_offsets_mapping=True)
            try:
                dict_answers = json.loads(sample['答案'])
                dict_answers_nodes = dict_answers['nodes']
                dict_answers_relations = dict_answers['relations']
            except:
                logger.info(f"数据格式错误: {dict_answers}")
                continue
            if dict_answers == '-1':
                continue
            TYPE = sample['类型']
            for i, (input_ids, token_type_ids, attention_mask, offset_mapping, overflow_to_sample_mapping) in enumerate(
                    zip(encodings['input_ids'],
                        encodings['token_type_ids'],
                        encodings['attention_mask'],
                        encodings['offset_mapping'],
                        encodings['overflow_to_sample_mapping'])):
                labels = []
                entity_type_indexes = set()
                id_2_type_token_index = self._create_id_2_token_index(dict_answers_nodes, offset_mapping)
                for dict_answers_node in dict_answers_nodes:
                    if dict_answers_node['id'] not in id_2_type_token_index:
                        continue
                    entity_type_indexes.add(id_2_type_token_index[dict_answers_node['id']])
                if TYPE == "NER":
                    '''实体识别
                    head0, head4, head5
                    '''
                    for dict_answers_node in dict_answers_nodes:
                        if dict_answers_node['id'] not in id_2_type_token_index:
                            continue
                        token_start_index, token_end_index = id_2_type_token_index[dict_answers_node['id']][1:]
                        type = dict_answers_node['type']
                        labels.append([1, token_start_index, token_end_index])
                        schema_label_index = self.schema['ner_2_index'][type]
                        labels.append([4, schema_label_index, token_start_index])
                        labels.append([5, schema_label_index, token_end_index])
                    ner_item = {'token': encodings.tokens(i),
                                'input_ids': input_ids,
                                'token_type_ids': token_type_ids,
                                'attention_mask': attention_mask,
                                'labels': labels,
                                'entity_type_indexes': entity_type_indexes,
                                'id': idx,
                                }
                    ner_data.append(ner_item)
                elif TYPE == 'ENTITY_SNT':
                    '''ent_snt训练
                    head4, head5
                    '''
                    for dict_answers_relation in dict_answers_relations:
                        node1_id = dict_answers_relation['node1']
                        if node1_id not in id_2_type_token_index:
                            continue
                        relation_type = dict_answers_relation['relation_type']
                        relation_value = dict_answers_relation['relation_value']
                        schema_label_index = self.schema['sentiment_2_index'][relation_value]
                        labels.append([4, schema_label_index, id_2_type_token_index[node1_id][1]])
                        labels.append([5, schema_label_index, id_2_type_token_index[node1_id][2]])
                    ent_snt_item = {'token': encodings.tokens(i),
                                    'input_ids': input_ids,
                                    'token_type_ids': token_type_ids,
                                    'attention_mask': attention_mask,
                                    'labels': labels,
                                    'entity_type_indexes': entity_type_indexes,
                                    'id': idx,
                                    }
                    ent_snt_data.append(ent_snt_item)
                elif TYPE == 'NER_RE':
                    '''re训练
                    head2, head3
                    '''
                    for dict_answers_relation in dict_answers_relations:
                        node1_id = dict_answers_relation['node1']
                        node2_id = dict_answers_relation.get('node2', None)
                        if node1_id not in id_2_type_token_index or node2_id not in id_2_type_token_index:
                            continue
                        # if id_2_type_token_index[node2_id][1] < id_2_type_token_index[node1_id][1]:
                        #     node1_id, node2_id = node2_id, node1_id
                        node1_token_start_index, node2_token_start_index = id_2_type_token_index[node1_id][1], \
                                                                           id_2_type_token_index[node2_id][1]
                        node1_token_end_index, node2_token_end_index = id_2_type_token_index[node1_id][2], \
                                                                       id_2_type_token_index[node2_id][2]
                        labels.append([2, node1_token_start_index, node2_token_start_index])
                        labels.append([3, node1_token_end_index, node2_token_end_index])
                    re_item = {'token': encodings.tokens(i),
                               'input_ids': input_ids,
                               'token_type_ids': token_type_ids,
                               'attention_mask': attention_mask,
                               'labels': labels,
                               'entity_type_indexes': entity_type_indexes,
                               'id': idx
                               }
                    re_data.append(re_item)
                elif TYPE == 'TRIPLE':
                    '''triplet训练
                    head1, head2, head3, head4, head5
                    '''
                    # 填充 head1
                    for dict_answers_node in dict_answers_nodes:
                        if dict_answers_node['id'] not in id_2_type_token_index:
                            continue
                        token_start_index, token_end_index = id_2_type_token_index[dict_answers_node['id']][1:]
                        type = dict_answers_node['type']
                        labels.append([1, token_start_index, token_end_index])
                        if dict_answers_node['type'] == '情感词':
                            schema_label_index = self.schema['tri_snt_2_index']['情感词']
                            labels.append([4, schema_label_index, token_start_index])
                            labels.append([5, schema_label_index, token_end_index])

                    # 填充 head3, head4, head5, head6
                    for dict_answers_relation in dict_answers_relations:
                        node1_id = dict_answers_relation['node1']
                        node2_id = dict_answers_relation.get('node2', None)
                        relation_type = dict_answers_relation['relation_type']
                        relation_value = dict_answers_relation['relation_value']
                        if relation_value.startswith('美妆日化plus-'):
                            relation_value = relation_value[9:]
                        if node1_id not in id_2_type_token_index:
                            continue
                        if node2_id is not None and node2_id not in id_2_type_token_index:
                            continue
                        if relation_type == '情感':
                            schema_label_index = self.schema['sentiment_2_index'][relation_value]
                            labels.append([4, schema_label_index, id_2_type_token_index[node1_id][1]])
                            labels.append([5, schema_label_index, id_2_type_token_index[node1_id][2]])
                        elif relation_type == '维度':
                            schema_label_index = self.schema['triplet_2_index'][relation_value]
                            labels.append([4, schema_label_index, id_2_type_token_index[node1_id][1]])
                            labels.append([5, schema_label_index, id_2_type_token_index[node1_id][2]])
                        elif relation_type == '相关':
                            # if id_2_type_token_index[node2_id][1] < id_2_type_token_index[node1_id][1]:
                            #     node1_id, node2_id = node2_id, node1_id
                            node1_token_start_index, node2_token_start_index = id_2_type_token_index[node1_id][1], \
                                                                               id_2_type_token_index[node2_id][1]
                            node1_token_end_index, node2_token_end_index = id_2_type_token_index[node1_id][2], \
                                                                           id_2_type_token_index[node2_id][2]
                            labels.append([2, node1_token_start_index, node2_token_start_index])
                            labels.append([3, node1_token_end_index, node2_token_end_index])
                            #
                            node1_type, node2_type = id_2_type_token_index[node1_id][0], \
                                                     id_2_type_token_index[node2_id][0]
                            entity_type_indexes.add((node1_type, node1_token_start_index, node1_token_end_index))
                            entity_type_indexes.add((node2_type, node2_token_start_index, node2_token_end_index))
                        else:
                            pass
                    triplet_item = {'token': encodings.tokens(i),
                                    'input_ids': input_ids,
                                    'token_type_ids': token_type_ids,
                                    'attention_mask': attention_mask,
                                    'labels': labels,
                                    'entity_type_indexes': entity_type_indexes,
                                    'id': idx
                                    }
                    triplet_data.append(triplet_item)
                else:
                    raise ValueError('Wrong 类型!')

        print(
            f"ner_data: {len(ner_data)}, ent_snt_data: {len(ent_snt_data)}, re_data: {len(re_data)}, triplet_data: {len(triplet_data)}")
        return ner_data, ent_snt_data, re_data, triplet_data

    def _create_node_type_id_for_re(self, node_type):
        if node_type == '品牌':
            return 0
        elif node_type == '品类':
            return 1
        elif node_type in ['产品名', '产品别名']:
            return 2
        elif node_type == '特征词':
            return 3
        elif node_type == '情感词':
            return 4
        else:
            return 5

    def _create_id_2_token_index(self, dict_answers_nodes, offset_mapping):
        id_2_type_token_index = {}
        for dict_answers_node in dict_answers_nodes:
            token_start_index, token_end_index = None, None
            type = dict_answers_node['type']
            ans_char_start_index, ans_char_end_index = dict_answers_node['start_index'], \
                                                       dict_answers_node['end_index']
            for index, (enc_char_start_index, enc_char_end_index) in enumerate(
                    offset_mapping[self.schema['schema_tokens_length']:]):
                if ans_char_start_index == enc_char_start_index:
                    token_start_index = index + self.schema['schema_tokens_length']
                if ans_char_end_index == enc_char_end_index:
                    token_end_index = index + self.schema['schema_tokens_length']
                if token_end_index is not None and token_start_index is not None:
                    id_2_type_token_index[dict_answers_node['id']] = (
                        type, token_start_index, token_end_index)
                    break
        return id_2_type_token_index

    def batch_loader(self, ner_data, ent_snt_data, re_data, triplet_data, batch_size):
        ner_dataset = self._build_dataset(ner_data, 'ner')
        ent_snt_dataset = self._build_dataset(ent_snt_data, 'ent_snt')
        re_dataset = self._build_dataset(re_data, 're')
        triplet_dataset = self._build_dataset(triplet_data, 'triplet')
        dataloaders = []
        for dataset in [ner_dataset, ent_snt_dataset, re_dataset, triplet_dataset]:
            dataloaders.append(DataLoader(dataset,
                                          batch_size,
                                          sampler=RandomSampler(dataset),
                                          collate_fn=pad_collate_fn,
                                          drop_last=False))

        return dataloaders

    def eval_batch_loader(self, data, max_seq_len=256, batch_size=128):
        dataset = self._build_dataset(data, 'eval')
        return DataLoader(dataset,
                          batch_size,
                          sampler=SequentialSampler(dataset),
                          collate_fn=pad_collate_fn,
                          drop_last=False)

    def _filter_unvalid_data(self, data):
        '''gu
        '''
        logger.info(f'The length of raw_data is {len(data)}')
        results = []
        for x in data:
            try:
                dict_answers = json.loads(x['答案'])
                dict_answers_nodes = dict_answers['nodes']
                dict_answers_relations = dict_answers['relations']
            except:
                logger.info(f"数据格式错误: {dict_answers}")
                continue
            if dict_answers == '-1':
                continue
            results.append(x)
        logger.info(f'After filter unvalid data, the length of data is {len(results)}')
        return results

    def _filter_data(self, data):
        '''过滤掉部分的数据，如关系识别，带有对象的
        '''
        logger.info(f'The length of raw_data is {len(data)}')
        results = []
        for x in data:
            types = [x['type'] for x in json.loads(x['答案'])['nodes']]
            if '对象' in types and x['类型'] == 'NER_RE':
                continue
            results.append(x)
        logger.info(f'After filter some data, the length of data is {len(results)}')
        return results

    def _padding(self, data, max_seq_len, val=0):
        res = []
        for seq in data:
            if len(seq) > max_seq_len:
                res.append(seq[:max_seq_len])
            else:
                res.append(seq + [val] * (max_seq_len - len(seq)))
        return res

    def _build_dataset(self, data, task_type):
        input_ids = torch.tensor(self._padding([item['input_ids'] for item in data], self.max_length), dtype=torch.long)
        token_type_ids = torch.tensor(self._padding([item['token_type_ids'] for item in data], self.max_length),
                                      dtype=torch.long)
        attention_mask = torch.tensor(self._padding([item['attention_mask'] for item in data], self.max_length),
                                      dtype=torch.long)
        if task_type == 'eval':
            dataset = TensorDataset(input_ids, token_type_ids, attention_mask)
            return dataset
        else:
            matrixs = []
            for index, item in enumerate(data):
                matrix = torch.full([6, self.max_length, self.max_length], fill_value=-1, dtype=torch.float32)
                labels = item['labels']
                entity_type_indexes = item['entity_type_indexes']
                if task_type == 'ner':
                    # head0
                    matrix[0] = torch.tril(matrix[0], diagonal=-1)
                    matrix[0, :self.schema['schema_tokens_length'], :] = -1
                    matrix[0, :, sum(item['attention_mask']):] = -1
                    # head4, head5
                    matrix[4:6, self.ner_index, self.schema_tokens_length:] = 0
                elif task_type == 'ent_snt':
                    # head4, head5
                    head_indexes = [label[2] for label in labels if label[0] == 4]
                    for head_index in head_indexes:
                        matrix[4, self.sentiment_index, head_index] = 0
                    tail_indexes = [label[2] for label in labels if label[0] == 5]
                    for tail_index in tail_indexes:
                        matrix[5, self.sentiment_index, tail_index] = 0
                elif task_type == 're':
                    # head2, head3
                    brand_indexes = list(set([x[1:] for x in entity_type_indexes if x[0] == '品牌']))
                    category_indexes = list(set([x[1:] for x in entity_type_indexes if x[0] == '品类']))
                    product_indexes = list(set([x[1:] for x in entity_type_indexes if x[0] in ['产品', '产品名', '产品别名']]))
                    other_indexes = list(set([x[1:] for x in entity_type_indexes if x[0] not in [
                        '品牌', '品类', '产品', '产品名', '产品别名', '特征词', '情感词']]))
                    for sub, obj in itertools.product(brand_indexes,
                                                      category_indexes + product_indexes + other_indexes):
                        matrix[2, sub[0], obj[0]] = 0
                        matrix[3, sub[1], obj[1]] = 0
                    for sub, obj in itertools.product(category_indexes,
                                                       product_indexes + other_indexes):
                        matrix[2, sub[0], obj[0]] = 0
                        matrix[3, sub[1], obj[1]] = 0

                    for sub, obj in itertools.product(product_indexes,
                                                      other_indexes):
                        matrix[2, sub[0], obj[0]] = 0
                        matrix[3, sub[1], obj[1]] = 0
                elif task_type == 'triplet':
                    # head0
                    matrix[1] = torch.tril(matrix[1], diagonal=-1)
                    matrix[1, :self.schema['schema_tokens_length'], :] = -1
                    matrix[1, :, sum(item['attention_mask']):] = -1
                    # head4, head5
                    matrix[4:6, self.triplet_index + self.tri_snt_index, self.schema_tokens_length:] = 0
                    # 情感词
                    head_indexes = [label[2] for label in labels if label[0] == 4 and label[1] in self.sentiment_index]
                    for head_index in head_indexes:
                        matrix[4, self.sentiment_index, head_index] = 0
                    tail_indexes = [label[2] for label in labels if label[0] == 5 and label[1] in self.sentiment_index]
                    for tail_index in tail_indexes:
                        matrix[5, self.sentiment_index, tail_index] = 0
                    # head2, head3
                    triplet_indexes = list(set([x[1:] for x in entity_type_indexes if x[0] == '特征词']))
                    tri_snt_indexes = list(set([x[1:] for x in entity_type_indexes if x[0] == '情感词']))
                    for sub, obj in itertools.product(triplet_indexes, tri_snt_indexes):
                        matrix[2, sub[0], obj[0]] = 0
                        matrix[3, sub[1], obj[1]] = 0

                for x1, x2, x3 in labels:
                    matrix[x1, x2, x3] = 1
                matrixs.append(matrix)

            matrixs = torch.stack(matrixs)
            dataset = TensorDataset(input_ids, token_type_ids, attention_mask, matrixs)

            return dataset
