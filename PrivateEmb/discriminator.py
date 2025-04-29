import tensorflow as tf
from exp_clip import expclip

class discriminator:
    def __init__(self, args, edge_distribution):
        with tf.variable_scope('forward_pass'):
            self.edge_distribution = edge_distribution
            self.u_i = tf.placeholder(name='u_i', dtype=tf.int32, shape=[None])
            self.u_j = tf.placeholder(name='u_j', dtype=tf.int32, shape=[None])
            self.label = tf.placeholder(name='label', dtype=tf.float32, shape=[None])
            self.optimal_val_ij = tf.placeholder(name='optimal_values', dtype=tf.float32,
                                                 shape=[None])
            self.fake_u_i_embedding = tf.placeholder(name='fake_u_i_emb', dtype=tf.float32,
                                                  shape=[None, args.embedding_dim])
            self.fake_u_j_embedding = tf.placeholder(name='fake_u_j_emb', dtype=tf.float32,
                                                  shape=[None, args.embedding_dim])
            self.Gaussian_noise_i = tf.placeholder(name='Gau_noise_i', dtype=tf.float32,
                                                  shape=[None, args.embedding_dim])
            self.Gaussian_noise_j = tf.placeholder(name='Gau_noise_j', dtype=tf.float32,
                                                  shape=[None, args.embedding_dim])

            self.embedding = tf.get_variable('target_emb', [args.num_of_nodes, args.embedding_dim],
                                             initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
            self.u_i_embedding = tf.matmul(tf.one_hot(self.u_i, depth=args.num_of_nodes), self.embedding)

            args.s = args.K + 2
            self.embedding = self.embedding / [tf.norm(self.embedding) * args.s]

            if args.proximity == 'first-order':
                self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.embedding)
            elif args.proximity == 'second-order':
                self.context_embedding = tf.get_variable('context_emb', [args.num_of_nodes, args.embedding_dim],
                                                         initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
                self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.context_embedding)

                self.context_embedding = self.context_embedding / [tf.norm(self.context_embedding) * args.s]

            self.inner_product = tf.reduce_sum(self.u_i_embedding * self.u_j_embedding, axis=1)

            sigmoid_test = tf.div(1.0, 1 + expclip(-self.label * self.inner_product,
                                                    args.low_bound, args.upper_bound))

            self.sgm_loss = -tf.log(sigmoid_test)
            # self.sgm_loss = -tf.reduce_mean(tf.log_sigmoid(self.label * self.inner_product))

            # ----------- adv term ------------
            self.adv_inner_product_1 = tf.reduce_sum(self.u_i_embedding * self.fake_u_i_embedding
                                                     + self.u_i_embedding * self.Gaussian_noise_i, axis=1)
            sigmoid_i = tf.div(1.0, 1 + expclip(-self.adv_inner_product_1,
                                                    args.low_bound, args.upper_bound))
            # sigmoid_i = tf.sigmoid(self.adv_inner_product_1)

            self.adv_inner_product_2 = tf.reduce_sum(self.u_j_embedding * self.fake_u_j_embedding
                                                     + self.u_j_embedding * self.Gaussian_noise_j, axis=1)
            sigmoid_j = tf.div(1.0, 1 + expclip(-self.adv_inner_product_2,
                                                    args.low_bound, args.upper_bound))
            # sigmoid_j = tf.sigmoid(self.adv_inner_product_2)

            self.weight_i = 1  # 0.5
            self.weight_j = 1  # 0.5

            # self.weight_i = tf.stop_gradient(1 / self.sigmoid_i)
            # self.weight_j = tf.stop_gradient(1 / self.sigmoid_j)

            # self.weight_i = 1 / self.sigmoid_i
            # self.weight_j = 1 / self.sigmoid_j

            self.adv_loss = -self.weight_i * tf.log(1 - sigmoid_i) - self.weight_j * tf.log(1 - sigmoid_j)

            self.loss = tf.reduce_mean(self.sgm_loss + self.adv_loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr_dis)
            self.train_op = self.optimizer.minimize(self.loss)
