import tensorflow as tf


class GraphGANOptimizer(object):

    def __init__(self, model, learning_rate=1e-3, feature_matching=True):
        self.la = tf.placeholder_with_default(1., shape=())

        with tf.name_scope('grad_penalty',
                           values=[model.logits_real, model.adjacency_tensor, model.edges_softmax,
                                   model.node_tensor, model.nodes_softmax, model.D_x, model.discriminator_units]):
            eps = tf.random_uniform(tf.shape(model.logits_real)[:1], dtype=model.logits_real.dtype)

            x_int0 = model.adjacency_tensor * tf.expand_dims(tf.expand_dims(tf.expand_dims(eps, -1), -1),
                                                             -1) + model.edges_softmax * (
                             1 - tf.expand_dims(tf.expand_dims(tf.expand_dims(eps, -1), -1), -1))
            x_int1 = model.node_tensor * tf.expand_dims(tf.expand_dims(eps, -1), -1) + model.nodes_softmax * (
                    1 - tf.expand_dims(tf.expand_dims(eps, -1), -1))

            grad0, grad1 = tf.gradients(model.D_x((x_int0, None, x_int1), model.discriminator_units), (x_int0, x_int1))

            self.grad_penalty = tf.reduce_mean(((1 - tf.norm(grad0, axis=-1)) ** 2), (-2, -1)) + tf.reduce_mean(
                ((1 - tf.norm(grad1, axis=-1)) ** 2), -1, keepdims=True)
            self.grad_penalty = tf.reduce_mean(self.grad_penalty)

        with tf.name_scope('loss_D',
                           values=[model.logits_real, model.logits_fake]):
            self.loss_D = - model.logits_real + model.logits_fake
            self.loss_D = tf.reduce_mean(self.loss_D)

        with tf.name_scope('loss_G',
                           values=[model.logits_fake]):
            self.loss_G = - model.logits_fake
            self.loss_G = tf.reduce_sum(self.loss_F) if feature_matching else tf.reduce_mean(self.loss_G)

        with tf.name_scope('loss_V',
                           values=[model.value_logits_real, model.rewardR, model.value_logits_fake, model.rewardF]):
            self.loss_V = (model.value_logits_real - model.rewardR) ** 2 + (
                    model.value_logits_fake - model.rewardF) ** 2
            self.loss_V = tf.reduce_mean(self.loss_V)

        with tf.name_scope('loss_RL',
                           values=[model.value_logits_fake]):
            self.loss_RL = - model.value_logits_fake
            self.loss_RL = tf.reduce_mean(self.loss_RL)

        with tf.name_scope('loss_F', values=[model.features_real, model.features_fake]):
            self.loss_F = (tf.reduce_mean(model.features_real, 0) - tf.reduce_mean(model.features_fake, 0)) ** 2

        alpha = tf.abs(tf.stop_gradient(self.loss_G / self.loss_RL))

        with tf.name_scope('train_step'):
            self.step_D = tf.Variable(0, trainable=False)
            self.train_step_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=self.loss_D + 10 * self.grad_penalty,
                global_step=self.step_D,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

            self.step_G = tf.Variable(0, trainable=False)
            self.train_step_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=tf.cond(tf.greater(self.la, 0), lambda: self.la * self.loss_G, lambda: 0.) + tf.cond(
                    tf.less(self.la, 1), lambda: (1 - self.la) * alpha * self.loss_RL, lambda: 0.),
                global_step=self.step_G,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

            self.train_step_V = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=self.loss_V,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='value'))

        self.summary_op_D = tf.summary.merge(
            [tf.summary.scalar(l, getattr(self, l)) for l in ('loss_D', 'grad_penalty')]
        )
        summary_G = tf.summary.scalar('loss_G', self.loss_G)
        summary_lam = tf.summary.scalar('lambda', self.la)
        self.summary_op_G = tf.summary.merge([summary_G, summary_lam])
        self.summary_op_RL = tf.summary.merge([
            summary_G, summary_lam,
            tf.summary.scalar('loss_V', self.loss_V),
            tf.summary.scalar('loss_RL', self.loss_RL),
        ])
