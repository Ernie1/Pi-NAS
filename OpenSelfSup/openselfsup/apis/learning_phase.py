# @author: RongcongChen <chenrc@mail2.sysu.edu.cn>
import os
import numpy as np
import lightgbm as lgb
import pandas as pd
import hyperopt
import logging

from sklearn.model_selection import train_test_split
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
from hyperopt.tpe import logger
from contextlib import redirect_stdout

def train_classifier(X, y):

    params = {
        "objective": "binary",
        "metric": "binary_error",
        "verbosity": -1,
        "seed": 0,
        "num_threads": 4
    } 

    X = pd.DataFrame(X)    
    y = pd.DataFrame(y)
     
    X_train, X_val, y_train, y_val = data_split(X, y, 0.2)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)
    
    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
        "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
        "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
        "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
        "reg_alpha": hp.uniform("reg_alpha", 0, 2),
        "reg_lambda": hp.uniform("reg_lambda", 0, 2),
        "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
    }

    def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, 100, 
                              valid_data, early_stopping_rounds=100, verbose_eval=0) 
            score = model.best_score["valid_0"][params["metric"]] 
            return {'loss': score, 'status': STATUS_OK}

    trials = Trials()
    with open(os.devnull, "w+") as file, redirect_stdout(file):
        logger.setLevel(logging.ERROR)
        best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                             algo=tpe.suggest, max_evals=10, verbose=0, 
                             rstate=np.random.RandomState(1)) 

        hyperparams = space_eval(space, best)
        model = lgb.train({**params, **hyperparams}, train_data, 100, valid_data, early_stopping_rounds=100, verbose_eval=0)
    return model
 
def data_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=1)

#You can modify this function to adjust the sampling ratio on the left and right sides.
def get_ucb(m, n_next, n_curt, threshold):
    m = m - threshold
    return m

class LearningPhase:
    def __init__(self, X, y, height, h):
        self.X = X
        self.y = y
        self.height = height
        self.h = h
        self.XSet = set()
        if self.h == 1:
            for i in range(X.shape[0]):
                s = '_'.join([str(j) for j in X[i]])
                self.XSet.add(s)
        self.m = np.median(y)
   
        self.threshold = np.median(y) 
        self.n = 0
        if self.height > 0:
            self.model =  train_classifier(X, y >= self.m)
            build_trial = 0
            while self.model.best_score["valid_0"]["binary_error"] >= 0.5 + max(0, (build_trial-100.0)/1000.0*0.3): 
                 build_trial += 1
                 self.model =  train_classifier(X, y >= self.m)
   
            self.path_list = []
            self.left = LearningPhase(X[y >= self.m], y[y >= self.m], self.height-1, self.h + 1)
            for p in self.left.path_list:
                self.path_list.append('l'+p) 
            self.right = LearningPhase(X[y < self.m], y[y < self.m], self.height-1, self.h + 1)
            for p in self.right.path_list:
                self.path_list.append('r'+p) 
            self.threshold = self.right.m 
        else:
            self.path_list = ['']


    def ucb_select(self):
        path_model = []
        c = self
        p_node = []
        while c.height != 0:
            path_model.append(c)
            c_w = 0.5 - c.model.best_score["valid_0"]["binary_error"]
            l_ucb = get_ucb(c.left.m, c.left.n, c.n, c.threshold)
            r_ucb = get_ucb(c.right.m, c.right.n, c.n, c.threshold)
            c = c.left if l_ucb >= r_ucb else c.right
            if l_ucb >= r_ucb:
                p_node.append('l')
            else:
                p_node.append('r')
        return path_model, p_node

    def constrain(self, path_model, path_node, x):
        x = pd.DataFrame(x.reshape([1, -1])) 
        for m, n in zip(path_model, path_node):
            predict = m.model.predict(x)
            if not ((n == 'l' and predict >= 0.5) or (n == 'r' and predict < 0.5)):
                return False
                 
        return True 
  
    def path(self, x):
        c = self
        p = []
        while c.height != 0:
            predict = c.model.predict(pd.DataFrame(x.reshape([1, -1])))
            if predict >= 0.5:
                p.append('l')
                c = c.left
            else:
                p.append('r')
                c = c.right
        return ''.join(p) 

    def sample(self, path_model, path_node, sampleFunction):
        sample_n = 200000
        i = 0
        path_n = ''.join(path_node)
        while i < sample_n:
            X_s = sampleFunction()
            if self.constrain(path_model, path_node, X_s):
                X_s_str = '_'.join([str(s) for s in X_s])
                if X_s_str not in self.XSet:
                    return X_s
           
            i += 1
        i = 0
        while i < sample_n:
            X_s = sampleFunction()
            X_s_str = '_'.join([str(s) for s in X_s])
            if X_s_str not in self.XSet:
                return X_s
        X_s = sampleFunction()
        return X_s
