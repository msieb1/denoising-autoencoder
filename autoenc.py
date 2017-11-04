# coding: utf8
"""
1. Download this gist.
2. Get the MNIST data.
    wget http://deeplearning.net/data/mnist/mnist.pkl.gz
3. Run this code.
    python autoencoder.py 100 -e 1 -b 20 -v
Wait about a minute ... and get a vialization of weights.
"""
import numpy
import argparse
import matplotlib.pyplot as plt
import math



class Autoencoder(object):
    def __init__(self, n_visible=784, n_hidden=784, W1=None, W2=None, b1=None, b2=None, noise=0.0, untied=False, dropout_rate=0.5):

        self.rng = numpy.random.RandomState(1)

        r = numpy.sqrt(6. / (n_hidden + n_visible + 1))

        if W1 == None:
            self.W1 = self.random_init(r, (n_hidden, n_visible))

        if W2 == None:
            if untied:
                W2 = self.random_init(r, (n_visible, n_hidden))
            else:
                W2 = self.W1.T

        self.W2 = W2

        if b1 == None:
            self.b1 = numpy.zeros((n_hidden,1))
        if b2 == None:
            self.b2 = numpy.zeros((n_visible,1))



        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.alpha = 0.1
        self.noise = noise
        self.untied = untied
        self.dropout_rate = dropout_rate

    def random_init(self, r, size):
        return numpy.array(self.rng.uniform(low=-r, high=r, size=size))

    def sigmoid(self, x):
        return 1. / (1. + numpy.exp(-x))

    def sigmoid_prime(self, x):
        return x * (1. - x)

    def corrupt(self, x, noise):
        return self.rng.binomial(size=x.shape, n=1, p=1.0 - noise) * x

    def encode(self, x):
        return self.sigmoid(numpy.dot(self.W1, x) + self.b1)

    def decode(self, y):
        return self.sigmoid(numpy.dot(self.W2, y) + self.b2)

    def get_cost(self, x, z):
        eps = 1e-10
        return - numpy.sum((x * numpy.log(z + eps) + (1. - x) * numpy.log(1. - z + eps)))

    def get_cost_and_grad(self, x_batch, dnum):

        cost = 0.
        grad_W1 = numpy.zeros(self.W1.shape)
        grad_W2 = numpy.zeros(self.W2.shape)
        grad_b1 = numpy.zeros(self.b1.shape)
        grad_b2 = numpy.zeros(self.b2.shape)

        for ii in range(0,(x_batch.shape[1])):
            x = numpy.atleast_2d(x_batch[:, ii]).T
            tilde_x = self.corrupt(x, self.noise)
            #tilde_x = x
            p = self.encode(tilde_x)
            m = (1-self.dropout_rate)*numpy.ones(p.shape) > numpy.random.rand(p.shape[0], p.shape[1])
            p *= m
            y = self.decode(p)

            cost += self.get_cost(x, y)

            delta1 = - (x - y)

            if self.untied:

                grad_W2 += numpy.outer(delta1, p)
            else:
                grad_W1 += numpy.outer(delta1, p).T

            grad_b2 += delta1

            delta2 = numpy.dot(self.W2.T, delta1) * self.sigmoid_prime(p)
            grad_W1 += numpy.outer(delta2, tilde_x)
            grad_b1 += delta2

        grad_W1 /= len(x_batch)
        grad_W2 /= len(x_batch)
        grad_b1 /= len(x_batch)
        grad_b2 /= len(x_batch)

        return cost, grad_W1, grad_W2, grad_b1, grad_b2

    def train(self, X, epochs=15, batch_size=20):


        ####### TRACK LOSS ##############
        loss_tracker = {'train': numpy.zeros((epochs+1, 1)), 'valid': numpy.zeros((epochs+1, 1))}
        loss_tracker['train'][0, 0] = 0
        loss_tracker['valid'][0, 0] = 0


        batch_num = len(X) / batch_size

        for epoch in range(epochs):
            total_cost = 0.0
            batch_num = int(batch_num)
            for i in range(batch_num):
                batch = X[:, i * batch_size: (i + 1) * batch_size]

                cost, gradW1, gradW2, gradb1, gradb2 = self.get_cost_and_grad(batch, len(X))

                total_cost += cost
                self.W1 -= self.alpha * gradW1
                self.W2 -= self.alpha * gradW2
                self.b1 -= self.alpha * gradb1
                self.b2 -= self.alpha * gradb2

                grad_sum = gradW1.sum() + gradW2.sum() + gradb1.sum() + gradb2.sum()

            total_cost /= len(X)
            print("Epoch %s: error is %s" % (epoch+1, total_cost))
            #update error and loss dictionaries
            loss_tracker['train'][epoch+1, 0] = total_cost
            p = self.encode(x_valid)
            y = self.decode(p)

            loss_tracker['valid'][epoch+1, 0] = self.get_cost(x_valid, y)/len(x_valid)

            ##################################

            #print("Reconstruction error: %s" % (numpy.sum(nump.abs(x_train - x_train_rec))))
            numpy.save('W_den100.npy', self.W1)
            #numpy.save('W2.npy', self.W2)
        return loss_tracker

    def visualize_weights_and_plot(self, path_to_save, loss_tracker, save_to_file=False):
        # 1. Plot feature map W
        path_to_file = '/home/max/PyCharm/PycharmProjects/10-707/hw2/'
        W = numpy.load(path_to_file + 'W_den100.npy')
        l = W.shape[0]
        W = W.reshape(l, 28, 28)
        r = int(math.sqrt(l))
        out = numpy.zeros([28 * r, 28 * r])
        for i in range(r):
            for j in range(r):
                out[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = W[i * r + j]
        plt.figure()
        plt.imshow(out, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        # plt.show()
        if save_to_file:
            name = 'Features_notdenAE100'
            path = path_to_save + name
            plt.savefig(path)

        # 2. Plot
        plt.figure(1)
        plt.clf()
        plt.plot(numpy.arange(1, epochs + 1), loss_tracker['train'][1:], 'ro-', markersize=4, label='train')
        plt.plot(numpy.arange(1, epochs + 1), loss_tracker['valid'][1:], 'yo-', markersize=4, label='valid')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        # plt.title('cross-entropy')
        plt.legend(loc='upper left')
        if save_to_file:
            name = 'graph_loss_denAE50'
            path = path_to_save + name
            plt.savefig(path)






x_train = (numpy.load('x_train.npy'))
y_train = (numpy.load('y_train.npy'))
x_valid = (numpy.load('x_valid.npy'))
y_valid = (numpy.load('y_valid.npy'))
x_test = (numpy.load('x_test.npy'))
y_test = (numpy.load('y_test.npy'))
# store data in a dictionary
data = {'x_train': x_train, 'y_train': y_train, 'x_valid': x_valid, 'y_valid': y_valid, 'x_test': x_test,
        'y_valid': y_valid}

##########SET HYPERPARAMETERS AND NET CONFIGURATION#######
epochs = 100
untied = False
n_hidden = 100
batch_size = 50
dropout_rate = 0.1
########################################################

ae = Autoencoder(n_hidden=n_hidden, noise=0, untied=untied)
try:
    loss_tracker = ae.train(x_train, epochs=epochs, batch_size=batch_size, dropout_rate=dropout_rate)
except KeyboardInterrupt:
    exit()
    pass


path = '/home/max/PyCharm/PycharmProjects/10-707/hw2/figures/AE/'
ae.visualize_weights_and_plot(path, loss_tracker, save_to_file=False)
plt.show()



