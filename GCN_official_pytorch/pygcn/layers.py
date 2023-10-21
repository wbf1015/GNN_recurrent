import math
import sys
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 这是一个可训练的参数矩阵，矩阵的大小就是in_features*outfeatures
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # 重置权重矩阵，特征的scale越大，那么权重值的可能范围就越小，用均匀分布来重置矩阵参数
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    # 这个好像和本身作者在论文里提的那种实现方式不太一样
    def forward(self, input, adj):
        # 这个方法用来进行矩阵乘法
        support = torch.mm(input, self.weight)
        # 这个适用于系数矩阵的乘法，具体的效果还是矩阵的乘法
        output = torch.spmm(adj, support)
        # print('self.weight.shape=',self.weight.shape)
        # print('input.shape=',input.shape)
        # print('adj.shape=',adj.shape)
        # print('support.shape=',support.shape)
        # print('output.shape=',output.shape)
        # sys.exit(-1)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    # 这个的作用就是返回一段字符串
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
