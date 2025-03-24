import tensorflow as tf
import numpy as np
import argparse
import networkx as nx
import graph_util
import math
from sklearn.externals import joblib
from rdp_accountant import compute_rdp
from rdp_accountant import get_privacy_spent
from generator import generator
from discriminator import discriminator
from CalcAUC import CalcAUC

parser = argparse.ArgumentParser()
parser.add_argument('--indep_run_times', default=5)
parser.add_argument('--n_epoch', default=50)
parser.add_argument('--embedding_dim', default=128, help='16 for node clustering, 128 for link prediction')
parser.add_argument('--d_batch_size', default=128, help='128 for node clustering, 512 for link prediction')
parser.add_argument('--g_batch_size', default=128, help='128 for node clustering, 512 for link prediction')
parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
parser.add_argument('--edge_sampling', default='numpy', help='numpy or atlas or uniform')
parser.add_argument('--node_sampling', default='uniform', help='numpy or atlas or uniform')
parser.add_argument('--d_epoch', default=15)
parser.add_argument('--g_epoch', default=5)
# parser.add_argument('--g_batch_size', default=512)
parser.add_argument('--lr_gen', default=0.1)
parser.add_argument('--lr_dis', default=0.1)
parser.add_argument('--K', default=5)
parser.add_argument('--low_bound', default=10**(-5), help='low bound for exp func')
parser.add_argument('--upper_bound', default=120, help='upper bound for exp func')
parser.add_argument('--dis_sigma', default=5)
parser.add_argument('--gen_sigma', default=5)
parser.add_argument('--delta', default=10**(-5))
parser.add_argument('--dis_sens', default=1, help='sensitivity for discriminator')
parser.add_argument('--epsilon', default=6)
parser.add_argument('--s', default=7, help='normalization weight parameter where s=k+2')

args = parser.parse_args()

class AliasSampling:
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res

class prepare_data:
    def __init__(self, graph_file=None):
        self.g = graph_file
        self.num_of_nodes = len(self.g.nodes())
        self.num_of_edges = len(self.g.edges())
        self.edges_raw = self.g.edges(data=True)
        self.nodes_raw = self.g.nodes(data=True)
        self.non_edges = list(nx.non_edges(self.g))

        self.edge_distribution = np.array([attr['weight'] for _, _, attr in self.edges_raw], dtype=np.float32)
        self.edge_distribution /= np.sum(self.edge_distribution)
        self.edge_sampling = AliasSampling(prob=self.edge_distribution)
        self.node_negative_distribution = np.power(
            np.array([self.g.degree(node, weight='weight') for node, _ in self.nodes_raw], dtype=np.float32), 0.75)
        self.node_negative_distribution /= np.sum(self.node_negative_distribution)
        self.node_sampling = AliasSampling(prob=self.node_negative_distribution)

        self.node_index = {}
        self.node_index_reversed = {}
        for index, (node, _) in enumerate(self.nodes_raw):
            self.node_index[node] = index
            self.node_index_reversed[index] = node
        self.edges = [(self.node_index[u], self.node_index[v]) for u, v, _ in self.edges_raw]

    def prepare_data_for_dis(self, sess, generator):
        global edge_batch_index, negative_node, node_batch_index
        if args.edge_sampling == 'numpy':
            edge_batch_index = np.random.choice(self.num_of_edges, size=args.d_batch_size, replace=False)
        elif args.edge_sampling == 'atlas':
            edge_batch_index = self.edge_sampling.sampling(args.d_batch_size)
        elif args.edge_sampling == 'uniform':
            edge_batch_index = np.random.randint(0, self.num_of_edges, size=args.d_batch_size)

        pos_u_i = []
        pos_u_j = []
        pos_label = []

        neg_u_i = []
        neg_u_j = []
        neg_label = []

        if args.edge_sampling == 'numpy':
            node_batch_index = np.random.choice(self.num_of_nodes, size=args.d_batch_size * args.K)

        count = 0
        for edge_index in edge_batch_index:
            edge = self.edges[edge_index]
            if self.g.__class__ == nx.Graph:
                if np.random.rand() > 0.5:
                    edge = (edge[1], edge[0])
            pos_u_i.append(edge[0])
            pos_u_j.append(edge[1])
            pos_label.append(1)

            for i in range(args.K):
                node_index = node_batch_index[count]
                if self.g.has_edge(self.node_index_reversed[edge[0]], self.node_index_reversed[node_index]):
                    neg_u_i.append(edge[0])
                    neg_u_j.append(node_index)
                    neg_label.append(1)
                else:
                    neg_u_i.append(edge[0])
                    neg_u_j.append(node_index)
                    neg_label.append(-1)

                count = count + 1

        return pos_u_i, pos_u_j, pos_label, neg_u_i, neg_u_j, neg_label

    def prepare_data_for_gen(self, sess, model, index, node_list, batch_size_g):
        node_ids = []
        if isinstance(node_list, list) is False:
            node_list = list(node_list)

        for node_id in node_list[index * batch_size_g: (index + 1) * batch_size_g]:
            node_ids.append(node_id)

        GaussianNoise_embedding = np.random.normal(loc=0, scale=args.gen_sigma, size=(len(node_ids), args.embedding_dim))

        node_i_embedding = sess.run(model.u_i_embedding, feed_dict={model.u_i: node_ids})
        node_j_embedding = sess.run(model.u_j_embedding, feed_dict={model.u_j: node_ids})

        return node_ids, GaussianNoise_embedding, node_i_embedding, node_j_embedding

class trainModel:
    def __init__(self, inf_display, graph, test_pos=None, test_neg=None, node_label=None):
        self.inf_display = inf_display
        self.node_label = node_label
        self.test_pos = test_pos
        self.test_neg = test_neg
        self.graph = graph
        self.data_loader = prepare_data(self.graph)
        args.num_of_nodes = self.data_loader.num_of_nodes
        args.num_of_edges = self.data_loader.num_of_edges
        self.model = discriminator(args, self.data_loader.edge_distribution)
        self.generator = generator(args)

    def train_dis(self, test_task=None, test_ratios=None, output_filename=None):
        print(args)
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
        for indep_run_time in range(args.indep_run_times):
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())  # note that this initilization's location
                flag_auc = 0
                # orders for RDP
                orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
                rdp = np.zeros_like(orders, dtype=float)

                gen_loss = 0.0
                gen_neg_loss = 0.0
                gen_cnt = 0
                found = False
                for epoch in range(args.n_epoch):
                    for d_epoch in range(args.d_epoch):
                        # for index in range(math.floor(args.num_of_edges / args.d_batch_size)):
                        pos_u_i, pos_u_j, pos_label, neg_u_i, neg_u_j, \
                        neg_label = self.data_loader.prepare_data_for_dis(sess, self.generator)

                        positive_sets = [pos_u_i, pos_u_j, pos_label]
                        negative_sets = [neg_u_i, neg_u_j, neg_label]

                        test_sets = [positive_sets, negative_sets]

                        count = 0
                        for current_set in test_sets:
                            u_i = current_set[0]
                            u_j = current_set[1]
                            label = current_set[2]

                            Gaussian_noise_i = np.random.normal(loc=0, scale=args.dis_sigma,
                                                                size=(len(u_i), args.embedding_dim))
                            Gaussian_noise_j = np.random.normal(loc=0, scale=args.dis_sigma,
                                                                size=(len(u_j), args.embedding_dim))

                            fake_u_i_embedding = sess.run(self.generator.gen_w_1, feed_dict={self.generator.node_ids: u_i})
                            fake_u_j_embedding = sess.run(self.generator.gen_w_2, feed_dict={self.generator.node_ids: u_j})

                            feed_dict = {self.model.u_i: u_i, self.model.u_j: u_j, self.model.label: label,
                                         self.model.fake_u_i_embedding: fake_u_i_embedding,
                                         self.model.fake_u_j_embedding: fake_u_j_embedding,
                                         self.model.Gaussian_noise_i: Gaussian_noise_i,
                                         self.model.Gaussian_noise_j: Gaussian_noise_j}
                            _, loss = sess.run([self.model.train_op, self.model.loss], feed_dict=feed_dict)

                            # --------- RDP mechanism -------------
                            number_of_edges = len(self.graph.edges())
                            number_of_nodes = len(self.graph.nodes())

                            if count == 0:
                                sampling_prob = args.d_batch_size / number_of_edges
                                # print('000')
                            else:
                                sampling_prob = args.d_batch_size * args.K / number_of_nodes
                                # print('111')

                            neg_sampling_prob = args.d_batch_size * args.K / number_of_nodes
                            steps = (d_epoch + 1) * (epoch + 1) * (count + 1)
                            rdp = compute_rdp(q=sampling_prob, noise_multiplier=args.dis_sigma, steps=steps, orders=orders)
                            _eps, _delta, _ = get_privacy_spent(orders, rdp, target_eps=args.epsilon)
                            # print(_eps, _delta)
                            if _delta > args.delta:
                                print('jump out')
                                found = True
                                break

                            count = count + 1
                        if found:
                            break
                    if found:
                        break

                    for g_epoch in range(args.g_epoch):
                        gen_loss, gen_neg_loss, gen_cnt = self.train_gen(sess, gen_loss, gen_neg_loss, gen_cnt)

                    # --------------------------------------------------------------------------------
                    embedding = sess.run(self.model.embedding)
                    if test_task == 'lp':
                        embedding_mat = np.dot(embedding, embedding.T)
                        auc = CalcAUC(embedding_mat, self.test_pos, self.test_neg)
                        if auc > flag_auc:
                            flag_auc = auc
                        print(indep_run_time, epoch, d_epoch, loss, auc)

    def train_gen(self, sess, gen_loss, neg_loss, gen_cnt):
        node_list = self.graph.nodes()
        batch_size_g = args.g_batch_size * (args.K + 1)
        for index in range(math.floor(len(node_list) / batch_size_g)):
            node_ids, noise_embedding, node_i_embedding, node_j_embedding \
                = self.data_loader.prepare_data_for_gen(sess, self.model, index, node_list, batch_size_g)
            _loss, _neg_loss = sess.run(
                [self.generator.g_updates, self.generator.loss],
                feed_dict={self.generator.node_ids: np.array(node_ids),
                           self.generator.noise_embedding: np.array(noise_embedding),
                           self.generator.pos_node_i_embedding: np.array(node_i_embedding),
                           self.generator.pos_node_j_embedding: np.array(node_j_embedding)})

        return gen_loss, neg_loss, gen_cnt

def compute_delta(steps, batch_size, dataset_size, target_eps):
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = batch_size / dataset_size
    # compute RDP of the Sampled Gaussian mechanism for each order
    rdp = compute_rdp(sampling_probability, args.dis_sigma, steps, orders)
    # compute epsilon given the list of RDP values and the target delta
    return get_privacy_spent(orders, rdp, target_eps=target_eps)

if __name__ == '__main__':
    test_task = 'lp'
    set_algo_name = 'AdvSGM'

    set_dataset_names = ['lp_facebook']
    set_split_name = 'train0.9_test0.1'
    for set_dataset_name in set_dataset_names:
        tf.reset_default_graph()
        set_epsilon_str = 'epsilon' + str(args.epsilon)
        set_learning_rate = 'step' + str(args.lr_dis)
        set_nepoch_name = 'nepoch' + str(args.n_epoch)
        # ------------------------------------------------------
        oriGraph_filename = '../data/' + set_dataset_name +'/train_1'
        train_filename = '../data/' + set_dataset_name + '/' + set_split_name + '/'

        # Load graph
        trainGraph = graph_util.loadGraphFromEdgeListTxt(oriGraph_filename, directed=False)
        trainGraph = nx.adjacency_matrix(trainGraph)
        # ------------------------------------------------------
        train_pos = joblib.load(train_filename + 'train_pos.pkl')
        train_neg = joblib.load(train_filename + 'train_neg.pkl')
        test_pos = joblib.load(train_filename + 'test_pos.pkl')
        test_neg = joblib.load(train_filename + 'test_neg.pkl')

        trainGraph = trainGraph.copy()  # the observed network
        trainGraph[test_pos[0], test_pos[1]] = 0  # mask test links
        trainGraph[test_pos[1], test_pos[0]] = 0  # mask test links
        trainGraph.eliminate_zeros()

        row, col = train_neg
        trainGraph = trainGraph.copy()
        trainGraph[row, col] = 1  # inject negative train
        trainGraph[col, row] = 1  # inject negative train
        trainGraph = nx.from_scipy_sparse_matrix(trainGraph)
        # ------------------------------------------------------
        print('Num nodes: %d, num edges: %d' % (trainGraph.number_of_nodes(), trainGraph.number_of_edges()))
        inf_display = [test_task, set_dataset_name]
        tm = trainModel(inf_display, trainGraph, test_pos=test_pos, test_neg=test_neg)
        tm.train_dis(test_task=test_task, output_filename=set_algo_name)
