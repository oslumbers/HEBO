import sys
sys.path.append('/home/oslumbers/Documents/git_repos/HEBO/HEBO')

import numpy as np
import pandas as pd
from typing import Callable
from sklearn.model_selection import cross_val_predict, KFold

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

import warnings
warnings.filterwarnings('ignore')

def tuner(
        model_class,
        space_config : [dict],
        X : np.ndarray,
        y : np.ndarray,
        metric : Callable,
        greater_is_better : bool = True,
        cv       = None,
        max_iter = 16,
        report   = False,
        hebo_cfg = None,
        verbose  = True,
        num_ens  = 1,
        ) -> (dict, pd.DataFrame):

    if hebo_cfg is None:
        hebo_cfg = {}
        hebo_cfg['num_ens'] = 2
    
    space = DesignSpace().parse(space_config)
    opt   = HEBO(space, **hebo_cfg)
    if cv is None:
        cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
    
    for i in range(max_iter):
        rec = opt.suggest()
        hyp = rec.iloc[0].to_dict()
        for k in hyp:
            if space.paras[k].is_numeric and space.paras[k].is_discrete:
                hyp[k] = int(hyp[k])
        model = model_class(**hyp)
        pred = cross_val_predict(model, X, y, cv = cv)
        score_v = metric(y, pred)
        sign = -1. if greater_is_better else 1
        opt.observe(rec, np.array([sign * score_v]))
        if verbose:
            print('Iter %d, best metric: %g' % (i, sign * opt.y.min()), flush = True)
    
    best_id = np.argmin(opt.y.reshape(-1))
    best_hyp = opt.X.iloc[best_id]
    df_report = opt.X.copy()
    df_report['metric'] = sign * opt.y
    if report:
        return best_hyp.to_dict(), df_report
    return best_hyp.to_dict()

if __name__ == '__main__':
    space_cfg = [
            {'name' : 'max_depth',        'type' : 'int', 'lb' : 1, 'ub' : 20},
            {'name' : 'min_samples_leaf', 'type' : 'pow', 'lb' : 1e-4, 'ub' : 0.5},
            {'name' : 'max_features',     'type' : 'cat', 'categories' : ['auto', 'sqrt', 'log2']},
            {'name' : 'bootstrap',        'type' : 'bool'},
            {'name' : 'min_impurity_decrease', 'type' : 'pow', 'lb' : 1e-4, 'ub' : 1.0},
            ]
    X, y = load_digits(return_X_y = True)
    result = tuner(RandomForestClassifier, space_cfg, X, y, metric = r2_score, max_iter = 7)
    print(result)