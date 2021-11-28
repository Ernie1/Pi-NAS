import numpy as np
import abc
from .learning_phase import LearningPhase
import torch
import torch.distributed as dist
import pickle as pkl
import os

class SearchPhase():
    # __metaclass__ = abc.ABCMeta
    def __init__(self, initial_samples=[], initial_sample=80, selects=10,
                 height_level=[400, 800, 1600, 3200], sample_func=None,
                 get_net_acc=None, logger=None, work_dir=None):
        self.sampleFunction = sample_func
        self.get_net_acc = get_net_acc
        self.logger = logger
        self.work_dir = work_dir
        network = self.sampleFunction()
        network = network.reshape([1, -1])
        self.X = np.delete(network, 0, 0)
        self.y = np.array([])
        assert len(initial_samples) <= initial_sample
        self.initial_samples = np.array(initial_samples)
        self.initial_sample = initial_sample
        self.selects = selects
        self.height_level = height_level
        self.current_select = 0
        self.net_acc = []
    # @abc.abstractmethod
    # def sampleFunction(self):
    #     return 
    # abstract method for inherit

    def selectSample(self):
        XSet = set()

        for X_s in self.initial_samples:
            X_s_str = '_'.join([str(s) for s in X_s])
            XSet.add(X_s_str)
            yield X_s

        while len(self.y) < self.initial_sample:
            X_s = self.sampleFunction()
            X_s_str = '_'.join([str(s) for s in X_s])
            while X_s_str in XSet:
                X_s = self.sampleFunction() 
                X_s_str = '_'.join([str(s) for s in X_s])
            XSet.add(X_s_str)
            yield X_s

        if dist.get_rank() == 0:
            self.classifier = LearningPhase(self.X, self.y, self.height(), 1)
        np.random.seed(len(self.y))

        while True:
            self.current_select = 0 
            while self.current_select < self.selects:
                if dist.get_rank() == 0:
                    self.path_model, self.path_node = self.classifier.ucb_select()
                    X_s = self.classifier.sample(self.path_model, self.path_node, self.sampleFunction)
                    X_s = torch.Tensor(X_s).cuda()
                else:
                    X_s = torch.zeros(self.X.shape[1]).cuda()
                dist.broadcast(X_s, 0)
                X_s = np.array(X_s.int().cpu(), dtype=int)
                np.random.seed(len(self.y))

                yield X_s
    
            if dist.get_rank() == 0:
                self.classifier = LearningPhase(self.X, self.y, self.height(), 1)
            np.random.seed(len(self.y))
   
    def height(self):
        l = len(self.y)
        return np.searchsorted(self.height_level, l) + 1

    def back_propagate(self, network, acc):
        if acc != None:
            network = network.reshape([1, -1])
            acc = np.array([acc])
            self.X = np.concatenate([self.X, network])
            self.y = np.concatenate([self.y, acc])
        if len(self.y) > self.initial_sample:
            if dist.get_rank() == 0:
                for n in self.path_model:
                    n.n = n.n + 1
            self.current_select += 1

    def step_start_trigger(self):
        # define your method before learning action space
        pass

    def step_end_trigger(self):
        if dist.get_rank() == 0 and len(self.y) % 100 == 0:
            self.save_net_acc()
    
    def run_end_trigger(self):
        if dist.get_rank() == 0:
            self.save_net_acc()

    def save_net_acc(self):
        with open(os.path.join(self.work_dir, 'Xy_%d.pkl' % len(self.y)), 'wb') as f:
            pkl.dump({'X':self.X, 'y':self.y}, f)

    def run(self, target_accuracy=1, max_samples=10000000):
        sample = self.selectSample()
        self.current_max_accuracy = 0 
        self.current_best_net = None
        while (self.current_max_accuracy < target_accuracy and len(self.y) < max_samples):
            self.step_start_trigger()
            network = next(sample) # Sampling a network for training
            if dist.get_rank() == 0:
                self.logger.info('enconding ({}): {}'.format(len(self.y)+1, network))
            if not isinstance(network, dict): 
                accuracy = self.get_net_acc(network)             # Get the accuracy of the sampling network after training
            else:
                raise NotImplementedError
            if dist.get_rank() == 0:
                self.logger.info('accuracy: {}\n'.format(accuracy))
            self.back_propagate(network, accuracy)          # update the learning phase according to the network and it's accuracy
            if accuracy > self.current_max_accuracy:
                self.current_max_accuracy = accuracy
                self.current_best_net = network 
            self.step_end_trigger()
        self.run_end_trigger()

    def get_top_accuracy(self, k):
        top_k= []
        top_k_index = np.argsort(self.y)[::-1][:k]
        for index in top_k_index:
            top_k.append([self.X[index],self.y[index]])
        return top_k
