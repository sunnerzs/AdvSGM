import numpy as np
import random
import scipy.sparse as ssp
import math
from sklearn import metrics
import networkx as nx
from PrivateEmb import graph_util
from sklearn.externals import joblib
import os

def sample_neg(net, test_ratio=0.2, train_pos=None, test_pos=None, max_train_num=None,
               all_unknown_as_negative=False):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None and test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None and train_pos is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num = len(train_pos[0]) if train_pos else 0
    test_num = len(test_pos[0]) if test_pos else 0
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    if not all_unknown_as_negative:
        # sample a portion unknown links as train_negs and test_negs (no overlap)
        while len(neg[0]) < train_num + test_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg = (neg[0][:train_num], neg[1][:train_num])
        test_neg = (neg[0][train_num:], neg[1][train_num:])
    else:
        # regard all unknown links as test_negs, sample a portion from them as train_negs
        while len(neg[0]) < train_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg = (neg[0], neg[1])
        test_neg_i, test_neg_j, _ = ssp.find(ssp.triu(net == 0, k=1))
        test_neg = (test_neg_i.tolist(), test_neg_j.tolist())
    return train_pos, train_neg, test_pos, test_neg

def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def generate_node2vec_embeddings(A, emd_size=128, negative_injection=False, train_neg=None):
    if negative_injection:
        row, col = train_neg
        A = A.copy()
        A[row, col] = 1  # inject negative train
        A[col, row] = 1  # inject negative train
    nx_G = nx.from_scipy_sparse_matrix(A)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)

def AA(A, test_pos, test_neg):
    # Adamic-Adar score
    A_ = A / np.log(A.sum(axis=1))
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    return CalcAUC(sim, test_pos, test_neg)

if __name__ == '__main__':
    # load data
    # train_pos = joblib.load('train_pos.pkl')

    train_filename = '../ProcessedData/lp_arxiv_snap/train_1'

    isDirected = False
    # Load graph
    trainGraph = graph_util.loadGraphFromEdgeListTxt(train_filename, directed=isDirected)
    # ------ compute average degree -----------
    d = dict(nx.degree(trainGraph))
    print('average degree', sum(d.values())/len(trainGraph.nodes()))
    print(trainGraph.number_of_nodes(), trainGraph.number_of_edges())
    # ------------------------------------------

    trainGraph = nx.adjacency_matrix(trainGraph)

    train_pos, train_neg, test_pos, test_neg = sample_neg(trainGraph, test_ratio=0.9, max_train_num=100000)

    # save train_pos, train_neg, test_pos, test_neg
    joblib.dump(train_pos, 'train_pos.pkl')
    joblib.dump(train_neg, 'train_neg.pkl')
    joblib.dump(test_pos, 'test_pos.pkl')
    joblib.dump(test_neg, 'test_neg.pkl')

    print('success')

    trainGraph = trainGraph.copy()  # the observed network
    trainGraph[test_pos[0], test_pos[1]] = 0  # mask test links
    trainGraph[test_pos[1], test_pos[0]] = 0  # mask test links
    trainGraph.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x

    row, col = train_neg
    trainGraph = trainGraph.copy()
    trainGraph[row, col] = 1  # inject negative train
    trainGraph[col, row] = 1  # inject negative train
    nx_G = nx.from_scipy_sparse_matrix(trainGraph)

    auc = AA(trainGraph, test_pos, test_neg)

    print('success')


