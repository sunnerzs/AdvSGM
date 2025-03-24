import tensorflow as tf

class generator:
    def __init__(self, args):
        self.gen_W_1 = tf.get_variable(name='gen_W_i', dtype=tf.float32,
                                       shape=[args.num_of_nodes, args.embedding_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        self.gen_B_1 = tf.get_variable(name='gen_B_i', dtype=tf.float32,
                                       shape=[args.embedding_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        self.gen_W_2 = tf.get_variable(name='gen_W_j', dtype=tf.float32,
                                       shape=[args.num_of_nodes, args.embedding_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        self.gen_B_2 = tf.get_variable(name='gen_B_j', dtype=tf.float32,
                                       shape=[args.embedding_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)

        self.gen_W_1 = self.gen_W_1 / [tf.norm(self.gen_W_1)]
        self.gen_W_2 = self.gen_W_2 / [tf.norm(self.gen_W_2)]

        self.node_ids = tf.placeholder(tf.int32, shape=[None])

        self.noise_embedding = tf.placeholder(tf.float32, shape=[None, args.embedding_dim])  # noise matrix

        self.pos_node_i_embedding = tf.placeholder(tf.float32, shape=[None, args.embedding_dim])  # pos samples from dis

        self.pos_node_j_embedding = tf.placeholder(tf.float32, shape=[None, args.embedding_dim])  # pos samples from dis

        self.gen_w_1 = tf.matmul(tf.one_hot(self.node_ids, depth=args.num_of_nodes), self.gen_W_1)

        self.gen_w_2 = tf.matmul(tf.one_hot(self.node_ids, depth=args.num_of_nodes), self.gen_W_2)

        test_1 = self.noise_embedding * self.gen_w_1 + self.gen_B_1

        # self.fake_embedding_1 = tf.nn.leaky_relu(test_1)

        self.fake_embedding_1 = tf.nn.sigmoid(test_1)

        test_2 = self.noise_embedding * self.gen_w_2 + self.gen_B_2

        # self.fake_embedding_2 = tf.nn.leaky_relu(test_2)

        self.fake_embedding_2 = tf.nn.sigmoid(test_2)

        self.gen_term_1 = self.fake_embedding_1 * self.pos_node_i_embedding \
                          + self.pos_node_i_embedding * self.noise_embedding
        self.gen_term_2 = self.fake_embedding_2 * self.pos_node_j_embedding \
                          + self.pos_node_j_embedding * self.noise_embedding
        self.loss = tf.reduce_mean(tf.log(1 - tf.log_sigmoid(self.gen_term_1)) + tf.log(1 - tf.log_sigmoid(self.gen_term_2)))

        self.optimizer = tf.train.AdamOptimizer(args.lr_gen)

        self.g_updates = self.optimizer.minimize(self.loss)