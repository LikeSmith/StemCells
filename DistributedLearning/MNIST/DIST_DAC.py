"""
DIST_CAC.py

Learning a model to classify MNIST images using a distributed algorithm that uses a Distributed Average Consensus
algorithm to share weights between nodes
"""

import numpy as np
import multiprocessing
import DistributedLearning.MNIST.MNIST as MNIST
import logging
import os

# Training Parameters
N_NODES = 4
N_CONN = 3
UPDATE_RATE = 10
VAL_SPLIT = 0.2
BATCH_SIZE = 128
N_EPOCHS = 5
RESULTS_DIR = 'results/'
RESULT_FILE_TEMPLATE = 'rslt-node%d.h5'
DATA_FILE_TEMPLATE = 'data-node%d.pkl'

# DAC Parameters
K_INIT = 1.0
Z_INIT = 0.0
GAMMA = 1.5
CONSENSUS_RATE = 0.01

# Separate Test and validation sets
perm = np.random.permutation(MNIST.IMG_TRAIN.shape[0])
val_size = int(MNIST.IMG_TRAIN.shape[0]*VAL_SPLIT)
IMG_VALID = MNIST.IMG_TRAIN[perm[:val_size]]
LAB_VALID = MNIST.LAB_TRAIN[perm[:val_size]]
IMG_TRAIN = MNIST.IMG_TRAIN[perm[val_size:]]
LAB_TRAIN = MNIST.LAB_TRAIN[perm[val_size:]]
IMG_EVAL = IMG_TRAIN[:BATCH_SIZE]
LAB_EVAL = LAB_TRAIN[:BATCH_SIZE]
N_BATCHES = int(np.ceil(IMG_TRAIN.shape[0]/BATCH_SIZE))

# Setup logging
logging.basicConfig(format='[%(levelname)s][%(acstime)s, Process: %(processname)s(%(process)d)]: %(message)s',
                    level=logging.DEBUG)


def gen_comm_graph(n_nodes, n_conn):
    edges = []
    while len(edges < n_conn):
        edge_i = (np.random.randint(len(n_nodes)), np.random.randint(len(n_nodes)))
        if edge_i[0] != edge_i[1] and edge_i not in edges:
            edges.append(edge_i)

    conn_pipes = [[]]*N_NODES
    for edge in edges:
        conn_a, conn_b = multiprocessing.Pipe()
        conn_pipes[edge[0]].append(conn_a)
        conn_pipes[edge[1]].append(conn_b)

    return conn_pipes


class Node(multiprocessing.Process):
    """
    Worker Class
    defines process that works together with other processes on a communication graph to train a network.
    """
    def __init__(self, conn, idx, **kwargs):
        super(Node, self).__init__(**kwargs)

        self.idx = idx
        self.conn = conn

    def run(self):
        logging.info('Starting...')

        hist = {'trn':{}, 'val':{}}

        # initialize model
        mdl = MNIST.MNIST_Model()
        mdl.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        mdl.predict(MNIST.IMG_TRAIN[0:1])

        for key in mdl.metrics_names():
            hist['trn'][key] = np.zeros(N_EPOCHS)
            hist['val'][key] = np.zeros(N_EPOCHS)

        # initialize DAC
        z = [np.ones_like(w)*Z_INIT for w in mdl.get_weights()]
        K = [[np.ones_like(w)*K_INIT for w in mdl.get_weights()] for _ in self.conn]

        # train network
        epoch = 0
        batch = 0
        perm = np.random.permutation(IMG_TRAIN.shape[0])

        while epoch < N_EPOCHS:
            # train network on some batches
            for _ in range(UPDATE_RATE):
                bat_img = IMG_TRAIN[perm[batch * BATCH_SIZE:min((batch + 1) * BATCH_SIZE, IMG_TRAIN.shape[0])]]
                bat_lab = LAB_TRAIN[perm[batch * BATCH_SIZE:min((batch + 1) * BATCH_SIZE, LAB_TRAIN.shape[0])]]
                outputs = mdl.train_on_batch(bat_img, bat_lab)

                for key, val in zip(mdl.metrics_names(), outputs):
                    hist['trn'][key][epoch] += val

                batch += 1

                if batch >= N_BATCHES:
                    epoch += 1
                    batch = 0
                    perm = np.random.permutation(IMG_TRAIN.shape[0])
                    break

            logging.debug('Block %d complete, syncing...'%np.ceil(batch/UPDATE_RATE))

            # share weights with neighbors
            weights_loc = mdl.get_weights()
            weights_rcv = []

            for conn in self.conn:
                conn.send(weights_loc)
            for conn in self.conn:
                weights_rcv.append(conn.recv())

            # update weights with DAC algorithm
            del_z = [np.zeros_like(w) for w in mdl.get_weights()]
            del_K = np.zeros(len(self.conn))

            for i in range(len(z)):
                del_z[i] += -GAMMA*z[i]
                for j in range(len(self.conn)):
                    del_z[i] += -K[j][i]*np.sign(weights_loc[i] - weights_rcv[j][i])
                    del_K[j][i] += np.abs(weights_loc[i] - weights_rcv[j][i])

                    K[j][i] += CONSENSUS_RATE*del_K[j][i]
                z[i] += CONSENSUS_RATE*del_z[i]

            mdl.set_weights([w+z_i for w, z_i in zip(weights_loc, z)])
