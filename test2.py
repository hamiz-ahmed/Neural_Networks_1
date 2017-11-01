import numpy as np
import pickle as cPickle
import os
import gzip
import time

def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = cPickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 1, 28, 28)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 1, 28, 28)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 1, 28, 28)
    train_y = train_y.astype('int32')
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    print('... done loading data')
    return rval


# start by defining simple helpers
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_d(x):
    # !! TODO ################
    return sigmoid(x) * (1.0 - sigmoid(x))
    # TODO ################


def tanh(x):
    return np.tanh(x)


def tanh_d(x):
    # !! TODO ################
    return 1 - (tanh(x) ** 2)
    # TODO ################


def relu(x):
    return np.maximum(0.0, x)


def relu_d(x):
    # !!TODO ################
    dx = np.zeros_like(x)
    dx[x > 0] = 1
    return dx
    # TODO ################


def softmax(x, axis=1):
    # to make the softmax a "safe" operation we will
    # first subtract the maximum along the specified axis
    # so that np.exp(x) does not blow up!
    # Note that this does not change the output.
    x_max = np.max(x, axis=axis, keepdims=True)
    x_safe = x - x_max
    e_x = np.exp(x_safe)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def unhot(one_hot_labels):
    """ Invert a one hot encoding, creating a flat vector """
    return np.argmax(one_hot_labels, axis=-1)


# then define an activation function class
class Activation(object):
    def __init__(self, tname):
        if tname == 'sigmoid':
            self.act = sigmoid
            self.act_d = sigmoid_d
        elif tname == 'tanh':
            self.act = tanh
            self.act_d = tanh_d
        elif tname == 'relu':
            self.act = relu
            self.act_d = relu_d
        else:
            raise ValueError('Invalid activation function.')

    def fprop(self, input):
        # we need to remember the last input
        # so that we can calculate the derivative with respect
        # to it later on
        self.last_input = input
        return self.act(input)

    def bprop(self, output_grad):
        return output_grad * self.act_d(self.last_input)


# define a base class for layers
class Layer(object):
    def fprop(self, input):
        """ Calculate layer output for given input
            (forward propagation).
        """
        raise NotImplementedError('This is an interface class, please use a derived instance')

    def bprop(self, output_grad):
        """ Calculate input gradient and gradient
            with respect to weights and bias (backpropagation).
        """
        raise NotImplementedError('This is an interface class, please use a derived instance')

    def output_size(self):
        """ Calculate size of this layer's output.
        input_shape[0] is the number of samples in the input.
        input_shape[1:] is the shape of the feature.
        """
        raise NotImplementedError('This is an interface class, please use a derived instance')


# define a base class for loss outputs
# an output layer can then simply be derived
# from both Layer and Loss
class Loss(object):
    def loss(self, output, output_net):
        """ Calculate mean loss given real output and network output. """
        raise NotImplementedError('This is an interface class, please use a derived instance')

    def input_grad(self, output, output_net):
        """ Calculate input gradient real output and network output. """
        raise NotImplementedError('This is an interface class, please use a derived instance')


# define a base class for parameterized things
class Parameterized(object):
    def params(self):
        """ Return parameters (by reference) """
        raise NotImplementedError('This is an interface class, please use a derived instance')

    def grad_params(self):
        """ Return accumulated gradient with respect to params. """
        raise NotImplementedError('This is an interface class, please use a derived instance')


# define a container for providing input to the network
class InputLayer(Layer):
    def __init__(self, input_shape):
        if not isinstance(input_shape, tuple):
            raise ValueError("InputLayer requires input_shape as a tuple")
        self.input_shape = input_shape

    def output_size(self):
        return self.input_shape

    def fprop(self, input):
        return input

    def bprop(self, output_grad):
        return output_grad


class FullyConnectedLayer(Layer, Parameterized):
    """ A standard fully connected hidden layer, as discussed in the lecture.
    """

    def __init__(self, input_layer, num_units,
                 init_stddev, activation_fun=Activation('relu')):
        self.num_units = num_units
        self.activation_fun = activation_fun
        # the input shape will be of size (batch_size, num_units_prev)
        # where num_units_prev is the number of units in the input
        # (previous) layer
        self.input_shape = input_layer.output_size()
        # !! TODO ################################
        # TODO: implement weight initialization
        # TODO ################################
        # this is the weight matrix it should have shape: (num_units_prev, num_units)
        weights_shape = (self.input_shape[1], self.num_units)
        self.W = np.random.standard_normal(weights_shape) * init_stddev  # FIXME
        # and this is the bias vector of shape: (num_units)
        bias_weight = (1, self.num_units)
        self.b = np.zeros(bias_weight)  # FIXME
        # create dummy variables for parameter gradients
        # no need to change these here!
        self.dW = None
        self.db = None

    def output_size(self):
        return (self.input_shape[0], self.num_units)

    def fprop(self, input):
        # !! TODO ################################################
        # TODO: implement forward propagation
        # NOTE: you should also handle the case were
        #       activation_fun is None (meaning no activation)
        #       then this is simply a linear layer
        # TODO ################################################
        # you again want to cache the last_input for the bprop
        # implementation below!
        self.last_input = input
        # FIXME
        activation = input.dot(self.W) + self.b
        if self.activation_fun == None:
            return activation
        else:
            return self.activation_fun.fprop(activation)
            # raise NotImplementedError("You should implement this!")
            # return np.zeros(self.output_size)

    def bprop(self, output_grad, ):
        """ Calculate input gradient (backpropagation). """
        # !! TODO ################################
        # TODO: implement backward propagation
        # TODO ###############################

        # HINT: you may have to divide the weights by n
        #       to make gradient checking work
        #       (since you want to divide the loss by number of inputs)
        n = output_grad.shape[0]
        # accumulate gradient wrt. the parameters first
        # we will need to store these to later update
        # the network after a few forward backward passes
        # the gradient wrt. W should be stored as self.dW
        # the gradient wrt. b should be stored as selfdb
        # raise NotImplementedError("you should implement this")
        self.dW = np.zeros((self.last_input.shape[1], self.num_units))  # FIXME
        self.db = np.zeros((1, self.num_units))  # FIXME
        # the gradient wrt. the input should be calculated here
        if self.activation_fun == None:
            da = output_grad
        else:
            da = self.activation_fun.bprop(output_grad)  # theta(L)/theta(act)
        z = self.last_input
        self.dW = (z.T.dot(da)) / n
        self.db = np.mean(da, axis=0)
        grad_input = np.dot(da, self.W.T)
        return grad_input

    def params(self):
        return self.W, self.b

    def grad_params(self):
        return self.dW, self.db


# finally we specify the interface for output layers
# which are layers that also have a loss function
# we will implement two output layers:
#  a Linear, and Softmax (Logistic Regression) layer
# The difference between output layers and and normal
# layers is that they will be called to compute the gradient
# of the loss through input_grad(). bprop will never
# be called on them!
class LinearOutput(Layer, Loss):
    """ A simple linear output layer that
        uses a squared loss (e.g. should be used for regression)
    """

    def __init__(self, input_layer):
        self.input_size = input_layer.output_size()

    def output_size(self):
        return (1,)

    def fprop(self, input):
        return input

    def bprop(self, output_grad):
        raise NotImplementedError(
            'LinearOutput should only be used as the last layer of a Network'
            + ' bprop() should thus never be called on it!'
        )

    def input_grad(self, Y, Y_pred):
        # !! TODO ####################################
        # TODO: implement gradient of squared loss
        # TODO ####################################
        loss_grad = np.abs(Y - Y_pred)
        return loss_grad  # FIXME-

    def loss(self, Y, Y_pred):
        loss = 0.5 * np.square(Y - Y_pred)
        return np.mean(np.sum(loss, axis=1))


class SoftmaxOutput(Layer, Loss):
    """ A softmax output layer that calculates
        the negative log likelihood as loss
        and should be used for classification.
    """

    def __init__(self, input_layer):
        self.input_size = input_layer.output_size()

    def output_size(self):
        return (1,)

    def fprop(self, input):
        return softmax(input)

    def bprop(self, output_grad):
        raise NotImplementedError(
            'SoftmaxOutput should only be used as the last layer of a Network'
            + ' bprop() should thus never be called on it!'
        )

    def input_grad(self, Y, Y_pred):
        # !! TODO #######################################################
        # TODO: implement gradient of the negative log likelihood loss
        # TODO #######################################################
        # HINT: since this would involve taking the log
        #       of the softmax (which is np.exp(x)/np.sum(x, axis=1))
        #       this gradient computation can be simplified a lot!
        loss_grad = -(Y - Y_pred)
        return loss_grad

    def loss(self, Y, Y_pred):
        # Assume one-hot encoding of Y
        # calculate softmax first
        # to make the loss numerically stable
        # you may want to add an epsilon in the log ;)
        eps = 1e-10
        # !! TODO ####################################
        # calculate negative log likelihood
        # TODO ####################################
        out = softmax(Y_pred)
        # TODO ####################################
        # calculate negative log likelihood
        # TODO #################################### -- done
        loss = -np.sum(Y * np.log(Y_pred + eps))
        return loss / Y.shape[0]


class NeuralNetwork:
    """ Our Neural Network container class.
    """

    def __init__(self, layers):
        self.layers = layers

    def _loss(self, X, Y):
        Y_pred = self.predict(X)
        return self.layers[-1].loss(Y, Y_pred)

    def predict(self, X):
        """ Calculate an output Y for the given input X. """
        # !! TODO ##########################################
        # TODO: implement forward pass through all layers
        # TODO ##########################################
        Y_pred = 0.0
        n = len(self.layers)
        inputX = X
        for i in range(n):
            inputX = self.layers[i].fprop(inputX)
        Y_pred = inputX
        return Y_pred

    def backpropagate(self, Y, Y_pred, upto=0):
        """ Backpropagation of partial derivatives through
            the complete network up to layer 'upto'
        """
        next_grad = self.layers[-1].input_grad(Y, Y_pred)
        # TODO ##########################################
        # TODO: implement forward pass through all layers
        # TODO ##########################################
        for layer in reversed(self.layers[upto:-1]):
            next_grad = layer.bprop(next_grad)
        return next_grad

    def classification_error(self, X, Y):
        """ Calculate error on the given data
            assuming they are classes that should be predicted.
        """
        Y_pred = unhot(self.predict(X))
        error = Y_pred != Y
        return np.mean(error)

    def sgd_epoch(self, X, Y, learning_rate, batch_size):
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        for b in range(n_batches):
            # !! TODO #####################################
            # Implement stochastic gradient descent here
            # TODO #####################################
            # start by extracting a batch from X and Y
            # (you can assume the inputs are already shuffled)

            # !! TODO: then forward and backward propagation + updates
            # HINT: layer.params() returns parameters *by reference*
            #       so you can easily update in-place
            inputX = X[b * batch_size:b * batch_size + batch_size]
            Y_true = Y[b * batch_size:b * batch_size + batch_size]
            Y_pred = self.predict(inputX)
            self.backpropagate(Y_true, Y_pred)
            for layer in self.layers:
                if isinstance(layer, Parameterized):
                    for param, grad in zip(layer.params(),
                                           layer.grad_params()):
                        param -= learning_rate * grad

    def gd_epoch(self, X, Y, learning_rate):
        # TODO ##################################################
        # Implement batch gradient descent here
        # A few hints:
        #   There are two strategies you can follow:
        #   Either shove the whole dataset throught the network
        #   at once (which can be problematic for large datasets)
        #   or run through it batch wise as in the sgd approach
        #   and accumulate the gradients for all parameters as
        #   you go through the data. Either way you should then
        #   do one gradient step after you went through the
        #   complete dataset!
        # TODO ##################################################
        self.sgd_epoch(X, Y, learning_rate, X.size[0])

    def train(self, X, Y, X_valid, Y_valid, learning_rate=0.1, max_epochs=100,
              batch_size=64, descent_type="sgd", y_one_hot=True):
        """ Train network on the given data. """
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        if y_one_hot:
            Y_train = one_hot(Y)
        else:
            Y_train = Y
        print("... starting training")
        for e in range(max_epochs + 1):
            if descent_type == "sgd":
                self.sgd_epoch(X_train, Y_train, learning_rate, batch_size)
            elif descent_type == "gd":
                self.gd_epoch(X_train, Y_train, learning_rate)
            else:
                raise NotImplementedError("Unknown gradient descent type {}".format(descent_type))

            # Output error on the training data
            train_loss = self._loss(X_train, Y_train)
            train_error = self.classification_error(X_train, Y)
            print('epoch {:.4f}, loss {:.4f}, train error {:.4f}'.format(e, train_loss, train_error))
            # TODO ##################################################
            # compute error on validation data:
            # simply make the function take validation data as input
            # and then compute errors here and print them
            # TODO ##################################################
            if (X_valid != None and Y_valid != None):
                valid_loss = self._loss(X_valid, Y_valid)
                valid_error = self.classification_error(X_valid, Y)
                print('epoch {:.4f}, loss {:.4f}, validation error {:.4f}'.format(e, valid_loss, valid_error))

    def check_gradients(self, X, Y):
        """ Helper function to test the parameter gradients for
        correctness. """
        for l, layer in enumerate(self.layers):
            if isinstance(layer, Parameterized):
                print('checking gradient for layer {}'.format(l))
                for p, param in enumerate(layer.params()):
                    # we iterate through all parameters
                    param_shape = param.shape

                    # define functions for conveniently swapping
                    # out parameters of this specific layer and
                    # computing loss and gradient with these
                    # changed parametrs
                    def output_given_params(param_new):
                        """ A function that will compute the output
                            of the network given a set of parameters
                        """
                        # copy provided parameters
                        param[:] = np.reshape(param_new, param_shape)
                        # return computed loss
                        return self._loss(X, Y)

                    def grad_given_params(param_new):
                        """A function that will compute the gradient
                           of the network given a set of parameters
                        """
                        # copy provided parameters
                        param[:] = np.reshape(param_new, param_shape)
                        # Forward propagation through the net
                        Y_pred = self.predict(X)
                        # Backpropagation of partial derivatives
                        self.backpropagate(Y, Y_pred, upto=l)
                        # return the computed gradient
                        return np.ravel(self.layers[l].grad_params()[p])

                    # let the initial parameters be the ones that
                    # are currently placed in the network and flatten them
                    # to a vector for convenient comparisons, printing etc.
                    param_init = np.ravel(np.copy(param))

                    # TODO ####################################
                    # TODO compute the gradient with respect to
                    #      the initial parameters in two ways:
                    #      1) with grad_given_params()
                    #      2) with finite differences
                    #         using output_given_params()
                    #         (as discussed in the lecture)
                    #      if your implementation is correct
                    #      both results should be epsilon close
                    #      to each other!
                    # TODO ####################################
                    epsilon = 1e-4
                    # making sure your gradient checking routine itself
                    # has no errors can be a bit tricky. To debug it
                    # you can "cheat" by using scipy which implements
                    # gradient checking exactly the way you should!
                    # To do that simply run the following here:
                    # import scipy.optimize
                    # err = scipy.optimize.check_grad(output_given_params,
                    #                                grad_given_params, param_init)
                    loss_base = output_given_params(param_init)
                    # TODO this should hold the gradient as calculated through bprop
                    gparam_bprop = grad_given_params(param_init)
                    # TODO this should hold the gradient calculated through
                    #      finite differences
                    gparam_fd = np.zeros_like(param_init)
                    for i in range(len(param_init)):
                        param_init[i] += epsilon
                        gparam_fd[i] = (output_given_params(param_init) - loss_base) / (epsilon)
                        param_init[i] -= epsilon
                    # calculate difference between them
                    err = np.mean(np.abs(gparam_bprop - gparam_fd))
                    print('diff {:.2e}'.format(err))
                    assert (err < epsilon)

                    # reset the parameters to their initial values
                    param[:] = np.reshape(param_init, param_shape)

input_shape = (5, 10)
n_labels = 6
layers = [InputLayer(input_shape)]

layers.append(FullyConnectedLayer(
                layers[-1],
                num_units=15,
                init_stddev=0.1,
                activation_fun=Activation('relu')
))
layers.append(FullyConnectedLayer(
                layers[-1],
                num_units=6,
                init_stddev=0.1,
                activation_fun=Activation('tanh')
))
layers.append(FullyConnectedLayer(
                layers[-1],
                num_units=n_labels,
                init_stddev=0.1,
                activation_fun=Activation('relu')
))
layers.append(SoftmaxOutput(layers[-1]))
nn = NeuralNetwork(layers)

# create random data
X = np.random.normal(size=input_shape)
# and random labels
Y = np.zeros((input_shape[0], n_labels))
for i in range(Y.shape[0]):
    idx = np.random.randint(n_labels)
    Y[i, idx] = 1.

nn.check_gradients(X, Y)

# load
Dtrain, Dval, Dtest = mnist()
X_train, y_train = Dtrain
# Downsample training data to make it a bit faster for testing this code
n_train_samples = 10000
train_idxs = np.random.permutation(X_train.shape[0])[:n_train_samples]
X_train = X_train[train_idxs]
y_train = y_train[train_idxs]

print("X_train shape: {}".format(np.shape(X_train)))
print("y_train shape: {}".format(np.shape(y_train)))

X_train = X_train.reshape(X_train.shape[0], -1)
print("Reshaped X_train size: {}".format(X_train.shape))

# Setup a small MLP / Neural Network
# we can set the first shape to None here to indicate that
# we will input a variable number inputs to the network
input_shape = (None, 28*28)
layers = [InputLayer(input_shape)]
layers.append(FullyConnectedLayer(
                layers[-1],
                num_units=100,
                init_stddev=0.01,
                activation_fun=Activation('relu')
))
layers.append(FullyConnectedLayer(
                layers[-1],
                num_units=100,
                init_stddev=0.01,
                activation_fun=Activation('relu')
))
layers.append(FullyConnectedLayer(
                layers[-1],
                num_units=10,
                init_stddev=0.01,
                # last layer has no nonlinearity
                # (softmax will be applied in the output layer)
                activation_fun=None
))
layers.append(SoftmaxOutput(layers[-1]))

nn = NeuralNetwork(layers)
# Train neural network
t0 = time.time()
nn.train(X_train, y_train, None, None,learning_rate=0.1,
         max_epochs=20, batch_size=64, y_one_hot=True)
t1 = time.time()
print('Duration: {:.1f}s'.format(t1-t0))