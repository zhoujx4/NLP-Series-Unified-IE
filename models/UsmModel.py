"""
@Time : 2023/4/2620:29
@Auth : zhoujx
@File ：UsmModel.py
@DESCRIPTION:

"""
import torch
from torch.nn import Module, BCEWithLogitsLoss
from transformers import AutoModel, BertModel

from models.layers import EfficientGlobalPointer
from utils.utils import RunningEMA, scatter_nd_pytorch


class UsmModel(Module):
    """
    """

    def __init__(self,
                 pretrain_model_path,
                 max_length,
                 schema,
                 with_adversarial_training=False,
                 matrix_hidden_size=400,
                 head_size=64,
                 dropout_rate=0.1,
                 ema_decay=0.999,
                 RoPE=True):
        super(UsmModel, self).__init__()
        self.encoder = BertModel.from_pretrained(pretrain_model_path)
        self.max_length = max_length
        self.schema = schema
        self.matrix_hidden_size = matrix_hidden_size
        self.head_size = head_size
        self.dropout_rate = dropout_rate
        self.RoPE = RoPE
        self.with_adversarial_training = with_adversarial_training
        self.dropout_layer = torch.nn.Dropout(p=self.dropout_rate, )
        self.global_pointer_layer = EfficientGlobalPointer(heads=6,
                                                           hidden_size=self.encoder.config.hidden_size,
                                                           head_size=self.head_size,
                                                           RoPE=self.RoPE,
                                                           use_bias=True,
                                                           tril_mask=False,
                                                           max_length=self.max_length)
        self.ner_index = [int(x) for x in self.schema['index_2_ner'].keys()]
        self.triplet_index = [int(x) for x in self.schema['index_2_triplet'].keys()]
        self.avg_metrics = [RunningEMA(ema_decay) for _ in range(4)]
        self.bce_loss = BCEWithLogitsLoss(reduction='none')

    def forward(self, batch_data, task_id):
        input_ids, token_type_ids, attention_mask, matrixs = batch_data
        # ner_train_data, ent_snt_train_data, re_train_data, triplet_train_data
        encoder_output = self.encoder(input_ids=input_ids,
                                      token_type_ids=token_type_ids,
                                      attention_mask=attention_mask)
        sequence_output, pooled_output = encoder_output[0], encoder_output[1]

        matrix_output = self.global_pointer_layer(sequence_output, attention_mask=attention_mask)  # [B, 3, L, L]
        loss = self.bce_loss(matrix_output, matrixs)

        mask = matrixs >= 0
        loss = torch.where(mask, loss, torch.zeros_like(loss))
        loss = loss.sum()
        self.avg_metrics[task_id].update(loss.item())
        loss = loss / mask.sum()

        return loss

    def get_res(self, batch_data):
        input_ids, token_type_ids, attention_mask = batch_data
        # ner_train_data, ent_snt_train_data, re_train_data, triplet_train_data
        encoder_output = self.encoder(input_ids=input_ids,
                                      token_type_ids=token_type_ids,
                                      attention_mask=attention_mask)
        sequence_output, pooled_output = encoder_output[0], encoder_output[1]

        matrix_output = self.global_pointer_layer(sequence_output, attention_mask=attention_mask)
        outputs_gt_0 = torch.argwhere(matrix_output > 0)

        return outputs_gt_0
