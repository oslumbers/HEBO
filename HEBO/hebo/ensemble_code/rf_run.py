import sys
sys.path.append('/home/oslumbers/Documents/git_repos/HEBO/HEBO')
import os

import numpy as np
import pandas as pd
import pickle
from typing import Callable, List, Tuple
import argparse

from sklearn.model_selection import cross_val_predict, KFold
from sklearn.datasets import load_digits, load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

import warnings
warnings.filterwarnings('ignore')

# Setup argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_ens', type=int, default=2)
parser.add_argument('--seeds', type=int, default=20)
parser.add_argument('--dataset', type=str, default='digits')

args = parser.parse_args()

def tuner(
        model_class,
        space_config : List[dict],
        X : np.ndarray,
        y : np.ndarray,
        metric : Callable,
        greater_is_better : bool = True,
        cv       = None,
        max_iter = 16,
        report   = True,
        hebo_cfg = None,
        verbose  = True,
        num_ens  = 2,
        num_suggestions = 4,
        num_rand_suggestions = 4,
        ):

    if hebo_cfg is None:
        hebo_cfg = {}
        hebo_cfg['num_ens'] = num_ens
        hebo_cfg['rand_sample'] = num_rand_suggestions
    
    space = DesignSpace().parse(space_config)
    opt   = HEBO(space, **hebo_cfg)
    if cv is None:
        cv = KFold(n_splits = 5, shuffle = True, random_state = 42)

    for i in range(max_iter):
        if i == 0:
            rec = opt.suggest(n_suggestions=(num_rand_suggestions * num_ens))
            for j in range(num_rand_suggestions * num_ens):
                ens_idx = j // num_rand_suggestions
                hyp = rec.iloc[j].to_dict()
                for k in hyp:
                    if space.paras[k].is_numeric and space.paras[k].is_discrete:
                        hyp[k] = int(hyp[k])
                model = model_class(**hyp)
                pred = cross_val_predict(model, X, y, cv = cv)
                score_v = metric(y, pred)
                sign = -1. if greater_is_better else 1
                opt.observe(rec.iloc[j], np.array([sign * score_v]), rand_samps=True, ens_idx=ens_idx)
            if verbose:
                print('Iter %d, best metric: %g' % (i, sign * np.min(opt.y)), flush = True)
        else:
            rec = opt.suggest(n_suggestions=num_suggestions)
            scores = np.zeros((num_suggestions, 1))
            for j in range(num_suggestions):
                hyp = rec.iloc[j].to_dict()
                for k in hyp:
                    if space.paras[k].is_numeric and space.paras[k].is_discrete:
                        hyp[k] = int(hyp[k])
                model = model_class(**hyp)
                pred = cross_val_predict(model, X, y, cv = cv)
                score_v = metric(y, pred)
                sign = -1. if greater_is_better else 1
                scores[j] = sign * score_v
            opt.observe(rec, scores, rand_samps=False)

            if verbose:
                print('Iter %d, best metric: %g' % (i, sign * np.min(opt.y)), flush = True)
    best_ids = np.argmin(opt.y, axis=1)

    best_hyp = [x.iloc[i] for i, x in zip(best_ids, opt.X)]
    df_report = opt.X.copy()
    for i in range(num_ens):
        df_report[i]['metric'] = sign * opt.y[i]
    if report:
        return [best_h.to_dict() for best_h in best_hyp], df_report
    return [best_h.to_dict() for best_h in best_hyp]

if __name__ == '__main__':
    seeds = args.seeds
    seed_res = []
    if args.dataset == 'digits':
        X, y = load_digits(return_X_y = True)
    elif args.dataset == 'iris':
        X, y = load_iris(return_X_y = True)
    elif args.dataset == 'wine':
        X, y = load_wine(return_X_y = True)
    elif args.dataset == 'breast_cancer':
        X, y = load_breast_cancer(return_X_y = True)

    for i in range(seeds):
        space_cfg = [
                {'name' : 'max_depth',         'type' : 'int', 'lb' : 1, 'ub' : 15},
                {'name' : 'max_features',      'type' : 'pow', 'lb' : 1e-2, 'ub' : 1.0},
                {'name' : 'min_samples_split', 'type' : 'pow', 'lb' : 1e-2, 'ub' : 1.0},
                {'name' : 'min_samples_leaf',  'type' : 'pow', 'lb' : 1e-2, 'ub' : 0.5},
                {'name' : 'min_weight_fraction_leaf', 'type' : 'pow', 'lb' : 1e-2, 'ub' : 0.5},
                {'name' : 'min_impurity_decrease', 'type' : 'num', 'lb' : 0.0, 'ub' : 0.5},
                ]
        result, report = tuner(RandomForestClassifier, space_cfg, X, y, metric = mean_squared_error, greater_is_better=False, max_iter = 3, num_ens=args.num_ens)
        print(f'result: {result}')
        seed_res.append(report)
    # Dump the report to a pickle file
    save_dir = f'results_testing/random_forest/{args.dataset}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, f'num_ens_{args.num_ens}.p'), 'wb') as f:
        pickle.dump(seed_res, f)
