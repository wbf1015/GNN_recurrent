import numpy as np
import scipy.sparse as sp
import torch
import sys


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    
    # 可以理解为一个这时一个二维数组，把.content中的内容原封不动的读出来了，没有做任何处理
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    # 构建一个稀疏矩阵用来存储每一篇文章是否包含某个单词 features.shape=(2708,1433)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 对标签做one-hot编码,生成一个[2708,7]的numpy数组
    labels = encode_onehot(idx_features_labels[:, -1])


    # START Build Graph
    # 下面两行代码创建了一个由节点标识(ID)映射到索引的一个字典
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    # 和之前一样，把.cites中的数据原封不动的加载进入内存
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    # 最终达到的效果就是把每一个在cites中出现的引用对儿的节点标识（id）全都换成索引
    # 具体的步骤就是先把饮用对拍平，然后map过去，最后再恢复成原来的大小
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # 创建一个稀疏矩阵保存所有的非零元素，矩阵的权重全部是1，矩阵的边由第二个参数表示（开头列表，结尾列表）
    # shape就是这个矩阵的大小是多少
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # 我觉得原来的注释不太准确，这一行的目的就是构造一个对称的邻接矩阵adj，因为之前构造的时候只构造了一半
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    # 将邻接矩阵加上单位矩阵再做归一化
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense())) #变成密集的矩阵
    labels = torch.LongTensor(np.where(labels)[1]) # 不要one-hot编码了，直接2708个数，每个数就代表对应下标索引的label
    adj = sparse_mx_to_torch_sparse_tensor(adj) #为了后面的tensor计算将sp格式的稀疏矩阵变成torch格式的
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) #拿到每一行的和
    r_inv = np.power(rowsum, -1).flatten() #对每一行的和取倒数，然后拍平成一维
    r_inv[np.isinf(r_inv)] = 0. # 当有一行都是零的时候，规定那一行的倒数也是0
    r_mat_inv = sp.diags(r_inv) # 把每一行和的倒数构造成一个对角矩阵
    mx = r_mat_inv.dot(mx) # 使用对角矩阵来计算行的归一化
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32) #将sp稀疏矩阵的格式转换为COO格式的稀疏矩阵
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)) # 这个张量表明了非零元素的坐标
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
