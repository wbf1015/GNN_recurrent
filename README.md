# GCN_recurrent



## GCN_official_pytorch 

该文件夹是论文作者对论文：SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS（ICLR2017）工作的官方代码。使用pytorch框架编写，作者在官方仓库中表明pytorch实现与tensorflow实现有细微差异，pytorch实现版本仅作为概念验证，而不为复现论文结果。

[[1609.02907\] Semi-Supervised Classification with Graph Convolutional Networks (arxiv.org)](https://arxiv.org/abs/1609.02907)

https://github.com/tkipf/pygcn

1、针对cora数据集，作者实验验证的是给出一些文献的互相引用关系（注意引用关系有重复），并且只给出一定文献的标签（分类）去预测其他文献的标签。

2、比较严谨的阅读者应该会从代码中发现稀疏的邻接矩阵adj的大小是13264，但是文献的个数是2708，并且引用关系是5429个，所以理论上稀疏的邻接矩阵应该有5429*2+2708=13566个非零元素，这和实验结果相比多出了302个，这是因为.cites文件中有151对重复的引用关系，我添加了check.py来证明冗余关系的存在。这也可以解释为什么在构造对称矩阵时作者需要在最后减去用原矩阵对应位置的元素，这是用来避免冗余的：

```python
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
```

