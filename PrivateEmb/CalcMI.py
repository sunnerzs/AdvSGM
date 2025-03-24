from sklearn import metrics
from sklearn.cluster import AffinityPropagation

def get_HT(X, Y, label_pred, name='k_means'):
    score_funcs = [
        metrics.adjusted_mutual_info_score,  # Adjusted Mutual Information
        metrics.mutual_info_score,  # Mutual Information
    ]

    km_scores_1 = metrics.mutual_info_score(Y, label_pred)
    km_scores_2 = metrics.adjusted_mutual_info_score(Y, label_pred)

    return km_scores_1, km_scores_2

def sortedDictValues1(adict):
    keys = list(adict.keys())
    keys.sort()
    values = list(map(adict.get, keys))
    return values

def evaluate(node_label, embedding_matrix):
    embedding_list = embedding_matrix.tolist()
    X = embedding_list
    # Sort the dictionary by the keys and then extract the corresponding values
    Y = sortedDictValues1(node_label)
    ap = AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15)
    ap.fit(X)
    label_pred = ap.labels_
    # print(label_pred)

    km_scores_1, km_scores_2 = get_HT(X, Y=Y, label_pred=label_pred, name='AffinityPropagation')
    return km_scores_1, km_scores_2