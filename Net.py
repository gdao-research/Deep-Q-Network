import tensorflow as tf
import numpy as np

def PReLU(x, alpha_name, name, init=0.001):
    with tf.variable_scope(name):
        _init = tf.constant_initializer(init)
        alpha = tf.get_variable(alpha_name, [], initializer=_init)
        x = tf.multiply(((1 + alpha)*x + (1 - alpha)*tf.abs(x)), 0.5)
    return x

class Net:
    def __init__(self, n_state, n_action, config):
        self.C = config
        self.n_state = n_state + [self.C['frame_stack']]
        self.n_action = n_action
        self.__make()

    def _make_ph(self):
        self.s_ph = tf.placeholder(tf.uint8, [None] + self.n_state, name='s_ph')
        self.a_ph = tf.placeholder(tf.int32, [None], name='a_ph')
        self.r_ph = tf.placeholder(tf.float32, [None], name='r_ph')
        self.d_ph = tf.placeholder(tf.float32, [None], name='d_ph')
        self.ns_ph = tf.placeholder(tf.uint8, [None] + self.n_state, name='ns_ph')
        self.lr_ph = tf.placeholder(tf.float32, [], name='lr_ph')

    def _build_net(self, inp, scope_name):
        inp = tf.cast(inp, tf.float32)/255.0
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            h = tf.layers.conv2d(inp, self.C['filter1'], self.C['size1'], self.C['strides1'], padding='same', kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0))
            h = PReLU(h, 'alpha', 'PReLU')
            h = tf.layers.conv2d(h, self.C['filter2'], self.C['size2'], self.C['strides2'], padding='same', kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0))
            h = PReLU(h, 'alpha_1', 'PReLU_1')
            h = tf.layers.conv2d(h, self.C['filter3'], self.C['size3'], self.C['strides3'], padding='same', kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0))
            h = PReLU(h, 'alpha_2', 'PReLU_2')
            h = tf.layers.dense(tf.layers.flatten(h), self.C['filter4'])
            h = tf.nn.leaky_relu(h, alpha=0.01, name='Leaky_ReLU')
            if self.C['dueling']:
                V = tf.layers.dense(h, 1)
                A = tf.layers.dense(h, self.n_action)
                Q = tf.add(A, V - tf.reduce_mean(A, 1, keepdims=True))
            else:
                Q = tf.layers.dense(h, self.n_action)
        return Q

    def _build_graph(self):
        pred = tf.reduce_sum(self.s_Q * tf.one_hot(self.a_ph, self.n_action), 1)
        if self.C['double']:
            self.ns_Q = self._build_net(self.ns_ph, 'online')
            choice = tf.argmax(self.ns_Q, 1)
            best_v = tf.reduce_sum(self.target_Q * tf.one_hot(choice, self.n_action), 1)
        else:
            best_v = tf.reduce_max(self.target_Q, 1)

        with tf.variable_scope('loss'):
            target = tf.clip_by_value(self.r_ph, -1, 1) + (1.-self.d_ph)*self.C['discount_factor']*tf.stop_gradient(best_v)
            loss = tf.losses.huber_loss(target, pred, reduction=tf.losses.Reduction.MEAN)
        return loss

    def _build_train(self):
        # optimizer = tf.train.AdamOptimizer(self.lr_ph)
        if self.C['optimizer'] == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(self.lr_ph, epsilon=1e-5)
        elif self.C['optimizer'] == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lr_ph)

        if self.C['grad_clip'] is None:
            self.optimize_op = optimizer.minimize(self.loss)
        else:
            grads = optimizer.compute_gradients(self.loss, var_list=self.sQ_global_params)
            for i, (grad, var) in enumerate(grads):
                if grad is not None:
                    grads[i] = (tf.clip_by_norm(grad, self.C['grad_clip']), var)
            self.optimize_op = optimizer.apply_gradients(grads)

        ops = []
        for o, t in zip(self.sQ_global_params, self.targetQ_global_params):
            ops.append(t.assign(o))
        self.update_target_op = tf.group(*ops, name='update_target')

    def __make(self):
        self._make_ph()
        self.s_Q = self._build_net(self.s_ph, 'online')
        self.target_Q = self._build_net(self.ns_ph, 'target')
        self.sQ_global_params = [p for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='online')]
        self.targetQ_global_params = [p for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')]
        self.loss = self._build_graph()
        self._build_train()

    def update_target(self):
        tf.get_default_session().run(self.update_target_op)

    def action(self, s):
        return np.argmax(tf.get_default_session().run(self.s_Q, feed_dict={self.s_ph:[s]})[0])

    def train(self, buffer, lr):
        sb, ab, rb, db, nsb = buffer.sample(self.C['batch_size'])
        tf.get_default_session().run(self.optimize_op, feed_dict={self.s_ph:sb, self.a_ph:ab, self.r_ph:rb, self.d_ph:db, self.ns_ph:nsb, self.lr_ph:lr})

    def save(self, directory):
        saver = tf.train.Saver()
        d = saver.save(tf.get_default_session(), directory+'/model.ckpt')

    def load(self, directory):
        saver = tf.train.Saver()
        saver.restore(tf.get_default_session(), directory+'/model.ckpt')

if __name__ == '__main__':
    from config import CONFIG as C
    sess = tf.InteractiveSession()
    nn = Net([84,84], 4, C)
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./test', sess.graph)
