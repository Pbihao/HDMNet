import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskMultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        super(MaskMultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None, cross=False):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat_interleave(self.head_num, 0)
        y, weights = self.scaled_dotproduct(q, k, v, mask, cross)
        y = self._reshape_from_batches(y)
        y = self.linear_o(y)
        return y, weights

    @staticmethod
    def gen_history_mask(x):
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )
        
    def scaled_dotproduct(self, query, key, value, mask=None, cross_att=False, tmp=0.1):
        assert (cross_att and mask is not None ) or (not cross_att and mask is None)
        dk = query.shape[-1]
        if not cross_att:
            scores = torch.einsum("bmc,bnc->bmn", query, key) / (math.sqrt(dk) + 1e-9)
        else:
            query, key = F.normalize(query, dim=2), F.normalize(key, dim=2)
            scores = torch.einsum("bmc,bnc->bmn", query, key) / tmp
        weight = scores

        if cross_att:
            attention = F.softmax(scores, dim=-2)
            attention = attention.masked_fill(mask == 0, 0)
            weight = (weight * mask).sum(2) / (mask.sum(2) + 1e-9)
        else:
            attention = F.softmax(scores, dim=-1)
            weight = weight.mean(2)

        if self.head_num > 1:
            weight = weight.reshape(weight.size(0) // self.head_num, self.head_num, weight.size(1))
            weight = weight.mean(1)
        
        return attention.matmul(value), weight