import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pickle
from tikzplotlib import save


## DEFINITION OF DGM NET USED IN THE ALGORITHM LATER
# LSTM-like layer used in DGM (see Figure 5.3 and set of equations on p. 45) - modification of Keras layer class

class LSTMLayer(tf.keras.layers.Layer):

    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, output_dim, input_dim, trans1="tanh", trans2="tanh"):
        '''
        Args:
            input_dim (int):       dimensionality of input data
            output_dim (int):      number of outputs for LSTM layers
            trans1, trans2 (str):  activation functions used inside the layer;
                                   one of: "tanh" (default), "relu" or "sigmoid"

        Returns: customized Keras layer object used as intermediate layers in DGM
        '''

        # create an instance of a Layer object (call initialize function of superclass of LSTMLayer)
        super(LSTMLayer, self).__init__()

        # add properties for layer including activation functions used inside the layer
        self.output_dim = output_dim
        self.input_dim = input_dim

        if trans1 == "tanh":
            self.trans1 = tf.nn.tanh
        elif trans1 == "relu":
            self.trans1 = tf.nn.relu
        elif trans1 == "sigmoid":
            self.trans1 = tf.nn.sigmoid

        if trans2 == "tanh":
            self.trans2 = tf.nn.tanh
        elif trans2 == "relu":
            self.trans2 = tf.nn.relu
        elif trans2 == "sigmoid":
            self.trans2 = tf.nn.relu

        ### define LSTM layer parameters (use Xavier initialization)
        # u vectors (weighting vectors for inputs original inputs x)
        self.Uz = self.add_variable("Uz", shape=[self.input_dim, self.output_dim],
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.Ug = self.add_variable("Ug", shape=[self.input_dim, self.output_dim],
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.Ur = self.add_variable("Ur", shape=[self.input_dim, self.output_dim],
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.Uh = self.add_variable("Uh", shape=[self.input_dim, self.output_dim],
                                    initializer=tf.contrib.layers.xavier_initializer())

        # w vectors (weighting vectors for output of previous layer)
        self.Wz = self.add_variable("Wz", shape=[self.output_dim, self.output_dim],
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.Wg = self.add_variable("Wg", shape=[self.output_dim, self.output_dim],
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.Wr = self.add_variable("Wr", shape=[self.output_dim, self.output_dim],
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.Wh = self.add_variable("Wh", shape=[self.output_dim, self.output_dim],
                                    initializer=tf.contrib.layers.xavier_initializer())

        # bias vectors
        self.bz = self.add_variable("bz", shape=[1, self.output_dim])
        self.bg = self.add_variable("bg", shape=[1, self.output_dim])
        self.br = self.add_variable("br", shape=[1, self.output_dim])
        self.bh = self.add_variable("bh", shape=[1, self.output_dim])

    # main function to be called
    def call(self, S, X):
        '''Compute output of a LSTMLayer for a given inputs S,X .

        Args:
            S: output of previous layer
            X: data input

        Returns: customized Keras layer object used as intermediate layers in DGM
        '''

        # compute components of LSTM layer output (note H uses a separate activation function)
        Z = self.trans1(tf.add(tf.add(tf.matmul(X, self.Uz), tf.matmul(S, self.Wz)), self.bz))
        G = self.trans1(tf.add(tf.add(tf.matmul(X, self.Ug), tf.matmul(S, self.Wg)), self.bg))
        R = self.trans1(tf.add(tf.add(tf.matmul(X, self.Ur), tf.matmul(S, self.Wr)), self.br))

        H = self.trans2(tf.add(tf.add(tf.matmul(X, self.Uh), tf.matmul(tf.multiply(S, R), self.Wh)), self.bh))

        # compute LSTM layer output
        S_new = tf.add(tf.multiply(tf.subtract(tf.ones_like(G), G), H), tf.multiply(Z, S))

        return S_new


# %% Fully connected (dense) layer - modification of Keras layer class

class DenseLayer(tf.keras.layers.Layer):

    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, output_dim, input_dim, transformation=None):
        '''
        Args:
            input_dim:       dimensionality of input data
            output_dim:      number of outputs for dense layer
            transformation:  activation function used inside the layer; using
                             None is equivalent to the identity map

        Returns: customized Keras (fully connected) layer object
        '''

        # create an instance of a Layer object (call initialize function of superclass of DenseLayer)
        super(DenseLayer, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        ### define dense layer parameters (use Xavier initialization)
        # w vectors (weighting vectors for output of previous layer)
        self.W = self.add_variable("W", shape=[self.input_dim, self.output_dim],
                                   initializer=tf.contrib.layers.xavier_initializer())

        # bias vectors
        self.b = self.add_variable("b", shape=[1, self.output_dim])

        if transformation:
            if transformation == "tanh":
                self.transformation = tf.tanh
            elif transformation == "relu":
                self.transformation = tf.nn.relu
        else:
            self.transformation = transformation

    # main function to be called
    def call(self, X):
        '''Compute output of a dense layer for a given input X
        Args:
            X: input to layer
        '''
        # compute dense layer output
        S = tf.add(tf.matmul(X, self.W), self.b)
        if self.transformation:
            S = self.transformation(S)
        return S


# %% Neural network architecture used in DGM - modification of Keras Model class

class DGMNet(tf.keras.Model):
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, layer_width, n_layers, input_dim, final_trans=None, scope='value', reuse=False):
        '''
        Args:
            layer_width:
            n_layers:    number of intermediate LSTM layers
            input_dim:   spaital dimension of input data (EXCLUDES time dimension)
            final_trans: transformation used in final layer

        Returns: customized Keras model object representing DGM neural network
        '''
        # create an instance of a Model object (call initialize function of superclass of DGMNet)
        super(DGMNet, self).__init__()
        with tf.variable_scope(scope, reuse=reuse):
            # define initial layer as fully connected
            # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
            self.initial_layer = DenseLayer(layer_width, input_dim + 1, transformation="tanh")
            # define intermediate LSTM layers
            self.n_layers = n_layers
            self.LSTMLayerList = []

            for _ in range(self.n_layers):
                self.LSTMLayerList.append(LSTMLayer(layer_width, input_dim + 1))
            # define final layer as fully connected with a single output (function value)
            self.final_layer = DenseLayer(1, layer_width, transformation=final_trans)

    # main function to be called
    def call(self, t, x):
        '''
        Args:
            t: sampled time inputs
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (t,x)
        '''
        # define input vector as time-space pairs
        X = tf.concat([t, x], 1)
        # call initial layer
        S = self.initial_layer.call(X)
        # call intermediate LSTM layers
        for i in range(self.n_layers):
            S = self.LSTMLayerList[i].call(S, X)
        # call final LSTM layers
        result = self.final_layer.call(S)

        return result


def perturbation_swirl(x, y, params):
    px = (params[0] * (y - params[2]) + np.finfo(float).eps) / tf.sqrt(
        tf.square(x - params[1]) + tf.square(y - params[2]) + np.finfo(float).eps)
    py = - (params[0] * (x - params[1]) + np.finfo(float).eps) / tf.sqrt(
        tf.square(x - params[1]) + tf.square(y - params[2]) + np.finfo(float).eps)
    return px, py


def perturbation_current_h(x, y, params):
    px = params[0] * tf.exp(-tf.square(y - params[1]) / (params[2] ** 2))
    py = tf.zeros_like(y)
    return px, py


def perturbation_current_v(x, y, params):
    px = tf.zeros_like(x)
    py = params[0] * tf.exp(-tf.square(x - params[1]) / (params[2] ** 2))
    return px, py


def perturbation_const(x, y, params):
    px = params[0] * tf.math.cos(params[1]) * tf.ones_like(x)
    py = params[0] * tf.math.sin(params[1]) * tf.ones_like(y)
    return px, py


def perturbation(pert_mode, x, y, pert_params):
    if pert_mode == 'swirl':
        return perturbation_swirl(x, y, params=pert_params)
    elif pert_mode == 'current_h':
        return perturbation_current_h(x, y, params=pert_params)
    elif pert_mode == 'current_v':
        return perturbation_current_v(x, y, params=pert_params)
    elif pert_mode == 'const':
        return perturbation_const(x, y, params=pert_params)
    elif pert_mode is None:
        return (tf.zeros_like(x), tf.zeros_like(y))
    else:
        raise RuntimeError('Perturbation mode not recognized')


def perturbation_swirl_np(x, y, params):
    px = (params[0] * (y - params[2]) + np.finfo(float).eps) / np.sqrt(
        np.square(x - params[1]) + np.square(y - params[2]) + np.finfo(float).eps)
    py = - (params[0] * (x - params[1]) + np.finfo(float).eps) / np.sqrt(
        np.square(x - params[1]) + np.square(y - params[2]) + np.finfo(float).eps)
    return px, py


def perturbation_current_h_np(x, y, params):
    px = params[0] * np.exp(-np.square(y - params[1]) / (params[2] ** 2))
    py = np.zeros_like(y)
    return px, py


def perturbation_current_v_np(x, y, params):
    px = np.zeros_like(x)
    py = params[0] * np.exp(-np.square(x - params[1]) / (params[2] ** 2))
    return px, py


def perturbation_const_np(x, y, params):
    px = params[0] * np.cos(params[1]) * np.ones_like(x)
    py = params[0] * np.sin(params[1]) * np.ones_like(y)
    return px, py


def perturbation_np(pert_mode, x, y, pert_params):
    if pert_mode == 'swirl':
        return perturbation_swirl_np(x, y, params=pert_params)
    elif pert_mode == 'current_h':
        return perturbation_current_h_np(x, y, params=pert_params)
    elif pert_mode == 'current_v':
        return perturbation_current_v_np(x, y, params=pert_params)
    elif pert_mode == 'const':
        return perturbation_const_np(x, y, params=pert_params)
    elif pert_mode is None:
        return (np.zeros_like(x), np.zeros_like(y))
    else:
        raise RuntimeError('Perturbation mode not recognized')


def sampler(nSim_interior, nSim_terminal, T, t_low, X_high, X_low, state_dim):
    # Sampler #1: domain interior
    t_interior = np.random.uniform(low=t_low - 0.1 * (T - t_low), high=T, size=[nSim_interior, 1])
    X_interior = np.random.uniform(low=X_low - 0.1 * (X_high - X_low), high=X_high + 0.1 * (X_high - X_low),
                                   size=[nSim_interior, state_dim])

    # Sampler #3: initial/terminal condition
    t_terminal = T * np.ones((nSim_terminal, 1))
    X_terminal = np.random.uniform(low=X_low - 0.1 * (X_high - X_low), high=X_high + 0.1 * (X_high - X_low),
                                   size=[nSim_terminal, state_dim])

    return t_interior, X_interior, t_terminal, X_terminal


# GLOBAL VALUES FOR MOVEMENT (NOTE THAT EXP INDICATOR IS USED FOR TRANSITION!)
f0 = 0.01
x99 = 1
a = f0 / (1 - f0)
k = - np.log(0.2 * a / (a + 0.8)) / (x99 ** 2)
kf = 1  # Max vel bounded by max(acc + pert) / kf --> No friction case: kf = 0
b = 2


def indicator_x(x):
    if x.ndim == 1:
        return a / (a + np.exp(-k * np.sum(np.square(x[0:2]))))
    else:
        return a / (a + np.exp(-k * np.sum(np.square(x[:, 0:2]), axis=1)))


def indicator_y(x):
    return indicator_x(x)


def running_cost(x):
    return np.sum(np.square(x[:, 0:2]), axis=1) / (np.sum(np.square(x[:, 0:2]), axis=1) + b)


def final_cost(x):
    return tf.expand_dims(tf.reduce_sum(tf.square(x[:, 0:2]), axis=1) / (tf.reduce_sum(tf.square(x[:, 0:2]), axis=1) + b), -1)  # Brake if x[:, 0:2], does not if x


def final_cost_np(x):
    if x.ndim == 1:
        val = np.sum(np.square(x[0:2])) / (np.sum(np.square(x[0:2])) + b)
    else:
        val = np.sum(np.square(x[:, 0:2]), axis=1) / (np.sum(np.square(x[:, 0:2]), axis=1) + b)
    return val


def loss(model, t_interior, X_interior, t_terminal, X_terminal, control, pert_mode=None, pert_params=None, kf=1):
    # Loss term #1: PDE
    # compute function value and derivatives at current sampled points
    V = model(t_interior, X_interior)
    V_t = tf.gradients(V, t_interior)[0]
    V_x = tf.gradients(V, X_interior)[0]
    indicator = a / (a + tf.math.exp(-k * tf.reduce_sum(tf.square(X_interior[:, 0:2]), axis=1)))
    px, py = perturbation(pert_mode, X_interior[:, 0], X_interior[:, 1], pert_params=pert_params)
    trans = tf.concat([tf.expand_dims(X_interior[:, 2] * indicator, -1),
                       tf.expand_dims(X_interior[:, 3] * indicator, -1),
                       tf.math.cos(control) + tf.expand_dims(px, -1) - kf * tf.expand_dims(X_interior[:, 2], -1),
                       tf.math.sin(control) + tf.expand_dims(py, -1) - kf * tf.expand_dims(X_interior[:, 3], -1)], 1)
    sum = tf.expand_dims(tf.reduce_sum(V_x * trans, axis=1), -1)
    running_cost = tf.expand_dims(tf.reduce_sum(tf.square(X_interior[:, 0:2]), axis=1) / (tf.reduce_sum(tf.square(X_interior[:, 0:2]), axis=1) + b), -1)
    diff_V = V_t + sum + running_cost  # Hamiltonian

    # compute average L2-norm of differential operator
    L1 = tf.reduce_mean(tf.square(diff_V))

    # Loss term #3: initial/terminal condition
    target_terminal = final_cost(X_terminal)
    fitted_terminal = model(t_terminal, X_terminal)

    L3 = tf.reduce_mean(tf.square(fitted_terminal - target_terminal))

    L_control = tf.reduce_mean(sum + running_cost)

    return L1, L3, L_control


def compute_fitted_optimal_control(model, state_tnsr, time_tnsr):
    V = model(time_tnsr, state_tnsr)
    V_x = tf.gradients(V, state_tnsr)[0]
    control = tf.math.atan2(-V_x[:, 3], -V_x[:, 2])

    return tf.expand_dims(control, -1)


class DGM(object):
    def __init__(self, T=1, state_dim=1, t_low=1e-10, state_low=-1, state_high=1, num_layers=3, nodes_per_layer=50,
                 pert_mode=None, pert_params=None, kf=1):
        #%% Parameters
        self.T = T    # terminal time
        self.state_dim = state_dim
        self.kf = kf
        # Solution parameters (domain on which to solve PDE)
        self.t_low = t_low
        self.state_low = state_low
        self.state_high = state_high

        # neural network parameters
        self.num_layers = num_layers  # Number of layers used in DGM model
        self.nodes_per_layer = nodes_per_layer  # Output dim of each layer

        # Perturbation parameters
        self.pert_mode = pert_mode
        self.pert_params = pert_params

        # initialize DGM model (last input: space dimension = 1)
        self.model = DGMNet(self.nodes_per_layer, self.num_layers, self.state_dim, scope='DGM_value_function')

        # tensor placeholders (_tnsr suffix indicates tensors)
        # inputs (time, space domain interior, space domain at initial time)
        self.t_interior_tnsr = tf.placeholder(tf.float32, [None, 1])
        self.state_interior_tnsr = tf.placeholder(tf.float32, [None, self.state_dim])
        self.t_terminal_tnsr = tf.placeholder(tf.float32, [None, 1])
        self.state_terminal_tnsr = tf.placeholder(tf.float32, [None, self.state_dim])

        # optimal control computed numerically from fitted value function
        self.control = compute_fitted_optimal_control(self.model, self.state_interior_tnsr, self.t_interior_tnsr)

        # loss
        self.L1_tnsr, self.L3_tnsr, self.L_control = loss(self.model, self.t_interior_tnsr,
                                                          self.state_interior_tnsr, self.t_terminal_tnsr,
                                                          self.state_terminal_tnsr, self.control, kf=self.kf)
        self.loss_tnsr = self.L1_tnsr + self.L3_tnsr

        # value function
        self.V = self.model(self.t_interior_tnsr, self.state_interior_tnsr)
        self.V_t = tf.gradients(self.V, self.t_interior_tnsr)[0]
        self.V_x = tf.gradients(self.V, self.state_interior_tnsr)[0]

        self.value_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DGM_value_function")
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss_tnsr, var_list=self.value_vars)

        # initialize variables
        init_op = tf.global_variables_initializer()

        # open session
        self.sess = tf.Session()
        self.sess.run(init_op)

    def load(self, filename):
        saver = tf.train.Saver(var_list=self.value_vars)
        saver.restore(self.sess, filename)

    def save(self, filename):
        saver = tf.train.Saver(var_list=self.value_vars)
        saver.save(self.sess, filename)

    def train(self, n_itr=250, steps_per_sample=10, batch_size=10000, nn_file=None, data_vals=None):

        # %% Train network
        # initialize loss per training
        loss_list = []
        l1_list = []
        l3_list = []
        print('TRAINING DGM')
        nSim_interior = nSim_terminal = batch_size
        # for each sampling stage
        for i in range(n_itr):

            # sample uniformly from the required regions
            t_interior, X_interior, t_terminal, X_terminal = sampler(nSim_interior, nSim_terminal, self.T, self.t_low,
                                                                     self.state_high, self.state_low, self.state_dim)

            # for a given sample, take the required number of SGD steps
            loss_aux = []
            l1_aux = []
            l3_aux = []
            for _ in range(steps_per_sample):
                loss, L1, L3, _ = self.sess.run([self.loss_tnsr, self.L1_tnsr, self.L3_tnsr, self.optimizer],
                                                feed_dict={self.t_interior_tnsr: t_interior,
                                                           self.state_interior_tnsr: X_interior,
                                                           self.t_terminal_tnsr: t_terminal,
                                                           self.state_terminal_tnsr: X_terminal})
                loss_aux.append(loss)
                l1_aux.append(L1)
                l3_aux.append(L3)
            loss_list.append(loss_aux)
            l1_list.append(l1_aux)
            l3_list.append(l3_aux)
            print('DGM: Iteration = ', i, '; Total loss = ', loss, '; Ham loss = ', L1, '; Final value loss = ', L3)

        if nn_file is not None:
            self.save(nn_file)

        if data_vals is not None:
            with open(data_vals, 'wb') as handle:
                pickle.dump({'loss': loss_list, 'l1': l1_list, 'l3': l3_list}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_training(self, data_vals, tikz_name=None, extended_plot=False, save_tex=False):

        if isinstance(data_vals, list):
            print('Opening the first element of the list...')
            data_vals = data_vals[0]
        with open(data_vals, 'rb') as handle:
            data = pickle.load(handle)

        for key in data.keys():
            mean = np.array([np.mean(a) for a in data[key]])
            # std = np.array([np.std(a) for a in data[key]])
            plt.semilogy(mean, label=str(key), alpha=0.5)
            # plt.fill_between(np.arange(mean.size), mean + std, mean - std, alpha=0.5)
        plt.xlabel('Training epoch')
        plt.ylabel('Losses')
        plt.legend(loc='best')
        if tikz_name is not None:
            save(tikz_name + '_dgm.tex')
        plt.show()

        if extended_plot:
            nps = 100
            npt = 3
            x = y = np.linspace(self.state_low[0], self.state_high[0], nps).reshape([nps, 1])
            t = np.linspace(self.t_low, self.T, npt).reshape([npt, 1])
            X, Y = np.meshgrid(x, y)
            states = np.vstack([np.ravel(X), np.ravel(Y), np.zeros_like(np.ravel(X)), np.zeros_like(np.ravel(Y))]).T
            for it in range(npt):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                v = self.sess.run([self.V], feed_dict={self.t_interior_tnsr: t[it] * np.ones((nps * nps, 1)),
                                                       self.state_interior_tnsr: states})[0]
                ax.plot_surface(X, Y, v.reshape(X.shape))
                ax.collections[0]._facecolors2d = ax.collections[0]._facecolors3d
                ax.collections[0]._edgecolors2d = ax.collections[0]._edgecolors3d
                ax.set_xlabel('x dim')
                ax.set_ylabel('y dim')
                ax.set_zlabel('value function')
                plt.title('value function, t='+str(t[it]))
                if save_tex:
                    pass
                    #plt.savefig(os.path.join(os.getcwd(), 'aux.png'))  # Needed to avoid tikz errors, this is weird
                    #save(tex_prefix + '_val_function_t_' + str(t[it]) + '.tex')
                plt.show()

            nps = 100
            npt = 5
            x = y = np.linspace(self.state_low[0], self.state_high[0], nps).reshape([nps, 1])
            t = np.linspace(self.t_low, self.T, npt).reshape([npt, 1])
            X, Y = np.meshgrid(x, y)
            states = np.vstack([np.ravel(X), np.ravel(Y), np.zeros_like(np.ravel(X)), np.zeros_like(np.ravel(Y))]).T
            for it in range(npt):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                v = self.sess.run([self.V_x], feed_dict={self.t_interior_tnsr: t[it] * np.ones((nps * nps, 1)),
                                                         self.state_interior_tnsr: states})[0]
                ax.plot_surface(X, Y, v[:, 2].reshape(X.shape), color='g', alpha=0.5)
                ax.plot_surface(X, Y, v[:, 3].reshape(X.shape), color='b', alpha=0.5)
                ax.set_xlabel('x dim')
                ax.set_ylabel('y dim')
                ax.set_zlabel('value function grad')
                plt.title('value function gradient, t=' + str(t[it]))
                plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            v = self.sess.run([self.V], feed_dict={self.t_interior_tnsr: self.T * np.ones((nps * nps, 1)),
                                                   self.state_interior_tnsr: states})[0]
            ax.plot_surface(X, Y, v.reshape(X.shape), color='b')
            ax.plot_surface(X, Y, final_cost_np(states).reshape(X.shape), color='r')
            ax.set_xlabel('x dim')
            ax.set_ylabel('y dim')
            ax.set_zlabel('value function')
            plt.title('value function final and desired')
            plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, final_cost_np(states).reshape(X.shape) - v.reshape(X.shape), color='b')
            ax.set_xlabel('x dim')
            ax.set_ylabel('y dim')
            ax.set_zlabel('value function')
            plt.title('value function final condition error')
            plt.show()

            nps = 20
            x = y = np.linspace(self.state_low[0], self.state_high[0], nps).reshape([nps, 1])
            X, Y = np.meshgrid(x, y)
            states = np.vstack([np.ravel(X), np.ravel(Y), np.zeros_like(np.ravel(X)), np.zeros_like(np.ravel(Y))]).T
            npt = 11
            t = np.linspace(self.t_low, self.T, npt).reshape([npt, 1])
            ys = [i + t + (i * t) ** 2 for i in range(npt)]
            cmap = cm.get_cmap('rainbow')
            colors = cmap(np.linspace(0, 1, len(ys)))
            fig = plt.figure()
            for it in range(npt):
                acs = self.sess.run([[self.control]], feed_dict={self.t_interior_tnsr: t[it] * np.ones((nps * nps, 1)),
                                                                 self.state_interior_tnsr: states})[0][0]
                if it == npt - 1:
                    plt.quiver(np.ravel(X), np.ravel(Y), np.cos(acs), np.sin(acs), label=str(t[it]), color=colors[it],
                               alpha=0.3)
                else:
                    plt.quiver(np.ravel(X), np.ravel(Y), np.cos(acs), np.sin(acs), label=str(t[it]), color=colors[it])
            plt.title('Optimal actions')
            plt.xlabel('x dim')
            plt.ylabel('y dim')
            plt.legend(loc='best')
            if save_tex:
                plt.savefig("file.pdf")  # Needed by save in order not to crash (weird bug)
                save(tex_prefix + '_optimal_policy.tex')
            plt.show()

            if self.pert_mode is not None:
                px, py = perturbation_np(self.pert_mode, np.ravel(X), np.ravel(Y), self.pert_params)
                plt.quiver(np.ravel(X), np.ravel(Y), px.reshape(X.shape) + np.finfo(float).eps,
                           py.reshape(Y.shape) + np.finfo(float).eps)
                plt.title('Perturbation field')
                plt.xlabel('x dim')
                plt.ylabel('y dim')
                if save_tex:
                    plt.savefig("file.pdf")  # Needed by save in order not to crash (weird bug)
                    save(tex_prefix + '_perturbation.tex')
                plt.show()

            def obtain_traj(initial_state, tv):
                dt = tv[1] - tv[0]
                states = [initial_state]
                states2 = [initial_state]
                controls = []
                controls2 = []
                for it in range(len(tv)):
                    # Optimal case
                    ac = self.sess.run([self.control], feed_dict={self.t_interior_tnsr: tv[it] * np.ones((1, 1)),
                                                                  self.state_interior_tnsr: states[-1].reshape(
                                                                      [1, self.state_dim])})[0][0][0]
                    px, py = perturbation_np(self.pert_mode, states[-1][0], states[-1][1], self.pert_params)
                    next_state = states[-1] + dt * np.array([indicator_x(states[-1]) * states[-1][2],
                                                             indicator_y(states[-1]) * states[-1][3],
                                                             np.cos(ac) + px - self.kf * states[-1][2],
                                                             np.sin(ac) + py - self.kf * states[-1][3]])
                    states.append(next_state)
                    controls.append(ac)
                    # Reference case
                    ac2 = np.arctan2(-states2[-1][1], -states2[-1][0])
                    px, py = perturbation_np(self.pert_mode, states2[-1][0], states2[-1][1], self.pert_params)
                    next_state2 = states2[-1] + dt * np.array([indicator_x(states2[-1]) * states2[-1][2],
                                                               indicator_y(states2[-1]) * states2[-1][3],
                                                               np.cos(ac2) + px - self.kf * states2[-1][2],
                                                               np.sin(ac2) + py - self.kf * states2[-1][3]])
                    states2.append(next_state2)
                    controls2.append(ac2)
                cost_optimal = running_cost(np.array(states))
                cost_optimal[-1] = final_cost_np(states[-1])  # Final cost!
                cost_optimal2 = running_cost(np.array(states2))
                cost_optimal2[-1] = final_cost_np(states2[-1])  # Final cost!
                return states, states2, controls, controls2, cost_optimal, cost_optimal2

            tv = np.linspace(self.t_low, self.T, 100)
            states, states2, controls, controls2, cost_optimal, cost_optimal2 = obtain_traj(np.array([5, 5, 0, 0]), tv)

            circle = plt.Circle((0,0), radius=x99, color='r')
            fig, ax = plt.subplots()
            ax.plot(np.array(states)[:, 0], np.array(states)[:, 1])
            ax.quiver(np.array(states)[1:, 0], np.array(states)[1:, 1], np.cos(controls), np.sin(controls),
                      color='g')  # Acceleration
            ax.quiver(np.array(states)[1:, 0], np.array(states)[1:, 1], np.array(states)[1:, 2], np.array(states)[1:, 3],
                      color='k')  # Velocity
            ax.plot(np.array(states)[:, 0], np.array(states)[:, 1], color='b')
            ax.plot(np.array(states2)[:, 0], np.array(states2)[:, 1], color='r')
            ax.add_artist(circle)
            nps = 20
            x = y = np.linspace(np.amin(np.array(states)), np.amax(np.array(states)), nps).reshape([nps, 1])
            X, Y = np.meshgrid(x, y)
            if self.pert_mode is not None:
                px, py = perturbation_np(self.pert_mode, np.ravel(X), np.ravel(Y), self.pert_params)
                plt.quiver(np.ravel(X), np.ravel(Y),
                           px.reshape(X.shape) + np.finfo(float).eps,
                           py.reshape(Y.shape) + np.finfo(float).eps)
            if save_tex:
                plt.savefig("file.pdf")  # Needed by save in order not to crash (weird bug)
                save(tex_prefix + '_traj_example.tex')
            fig.show()

            # Cost plot

            plt.plot(cost_optimal, 'b')
            plt.plot(cost_optimal2, 'r')
            plt.title('Cost obtained')
            if save_tex:
                save(tex_prefix + '_cost.tex')
            plt.show()
            print('Total cost optimal = ', np.sum(cost_optimal), '; cost reference = ', np.sum(cost_optimal2))

            #Cost comparison
            n_trajs = 25
            copt = np.zeros(n_trajs)
            copt2 = np.zeros(n_trajs)
            so = []
            sc = []
            for traj in range(n_trajs):
                s1, s2, _, _, cost_optimal, cost_optimal2 = obtain_traj(
                    (self.state_high - self.state_low) * np.random.rand(self.state_dim) + self.state_low, tv)
                so.append(s1)
                sc.append(s2)
                copt[traj] = np.sum(cost_optimal)
                copt2[traj] = np.sum(cost_optimal2)
            plt.plot(copt2 - copt, 'b')
            plt.plot(np.zeros(n_trajs), 'r')
            plt.xlabel('Trajectory')
            plt.ylabel('Cost gain')
            plt.title('Cost gain on randomly initialized trajectories')
            if save_tex:
                save(tex_prefix + '_cost_gain.tex')
            plt.show()

            circle = plt.Circle((0, 0), radius=x99, color='r')
            fig, ax = plt.subplots()
            for traj in range(n_trajs):
                ax.plot(np.array(so[traj])[:, 0], np.array(so[traj])[:, 1], 'b')
                ax.plot(np.array(sc[traj])[:, 0], np.array(sc[traj])[:, 1], 'r')
            ax.add_artist(circle)
            nps = 20
            x = y = np.linspace(-1, 1, nps).reshape([nps, 1])
            X, Y = np.meshgrid(x, y)
            if self.pert_mode is not None:
                px, py = perturbation_np(self.pert_mode, np.ravel(X), np.ravel(Y), self.pert_params)
                plt.quiver(np.ravel(X), np.ravel(Y),
                           px.reshape(X.shape) + np.finfo(float).eps,
                           py.reshape(Y.shape) + np.finfo(float).eps)
            if save_tex:
                plt.savefig("file.pdf")  # Needed by save in order not to crash (weird bug)
                save(tex_prefix + '_trajs.tex')
            fig.show()

    def test(self, env, discount, nn_name=None, n_episodes=100, initial_states=None):
        # Returns a list of dictionaries with the data of several runs
        if nn_name is not None:
            self.load(nn_name)  # To load a saved NN

        if initial_states is not None:
            assert len(initial_states) == n_episodes

        output_data = []

        for e in range(n_episodes):
            if initial_states is not None:
                state, done = env.reset(initial_states[e]), False
            else:
                state, done = env.reset(), False

            episode_data = {'states': [state],
                            'actions': [],
                            'rewards': [],
                            'episode_reward': 0,
                            'episode_reward_disc': 0,
                            'episode_timestaps': 0}
            tv = np.arange(self.t_low, env.time_step * env.n_max, env.time_step).astype(list)

            for t in tv:
                episode_data['episode_timestaps'] += 1
                action = self.sess.run([self.control], feed_dict={self.t_interior_tnsr: t * np.ones((1, 1)),
                                                                  self.state_interior_tnsr: state})[0][0][0]
                action = np.squeeze(np.array([np.cos(action), np.sin(action)]))
                # Perform action
                next_state, reward, done, _ = env.step(action)
                state = next_state

                # Save data
                episode_data['states'].append(state)
                episode_data['actions'].append(action)
                episode_data['rewards'].append(reward)
                episode_data['episode_reward'] += reward
                episode_data['episode_reward_disc'] += discount ** (episode_data['episode_timestaps'] - 1) * reward

                if done:
                    break

            output_data.append(episode_data)
        return output_data

    def plot_policy(self, env, nn_name=None, norm_value=False):
        if nn_name is not None:
            self.load(nn_name)  # To load a saved NN

        npp = 20  # Points to represent the policy
        x = y = np.linspace(env.observation_space.low[0], env.observation_space.high[0], npp)
        X, Y = np.meshgrid(x, y)
        observations = np.vstack([np.ravel(X), np.ravel(Y), np.zeros(npp ** 2), np.zeros(npp ** 2)]).T

        acs = self.sess.run([[self.control]], feed_dict={self.t_interior_tnsr: np.zeros((npp * npp, 1)),
                                                         self.state_interior_tnsr: observations})[0][0]
        plt.quiver(np.ravel(X), np.ravel(Y), np.cos(acs), np.sin(acs))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('DGM policy')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        v = self.sess.run([self.V], feed_dict={self.t_interior_tnsr: self.T * np.ones((npp * npp, 1)),
                                               self.state_interior_tnsr: observations})[0]
        if norm_value:
            # Normalize v between 0 and 1
            aux = v - np.amin(v)
            v = aux / np.amax(aux)
        ax.plot_surface(X, Y, v.reshape(X.shape), color='b')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.title('DGM value function estimated')
        plt.show()

