# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


import sys
from typing import Optional

import numpy  as np
import pandas as pd
import torch
from copy import deepcopy
from torch.quasirandom import SobolEngine
from sklearn.preprocessing import power_transform

from hebo.design_space.design_space import DesignSpace
from hebo.models.model_factory import get_model, get_ensemble_models
from hebo.acquisitions.acq import MACEEns, MeanEns, SigmaEns
from hebo.acq_optimizers.evolution_optimizer import EvolutionOpt

from .abstract_optimizer import AbstractOptimizer

torch.set_num_threads(min(1, torch.get_num_threads()))

class HEBO(AbstractOptimizer):
    support_parallel_opt  = True
    support_combinatorial = True
    support_contextual    = True
    def __init__(self, space, num_ens=1, model_name = 'gpy', rand_sample = None, acq_cls = MACEEns, es = 'nsga2', model_config = None,
                 scramble_seed: Optional[int] = None ):
        """
        model_name  : surrogate model to be used
        rand_sample : iterations to perform random sampling
        scramble_seed : seed used for the sobol sampling of the first initial points
        """
        super().__init__(space)
        self.space       = space
        self.es          = es
        self.X           = [pd.DataFrame(columns = self.space.para_names) for _ in range(num_ens)]
        self.y           = [np.zeros((0, 1)) for _ in range(num_ens)]
        self.model_name  = model_name
        self.rand_sample = 1 + self.space.num_paras if rand_sample is None else max(2, rand_sample)
        self.scramble_seed = scramble_seed
        self.sobol       = SobolEngine(self.space.num_paras, scramble = True, seed = scramble_seed)
        self.acq_cls     = acq_cls
        self.num_ens     = num_ens
        self._model_config = model_config

    def quasi_sample(self, n, fix_input = None): 
        samp    = self.sobol.draw(n)
        samp    = samp * (self.space.opt_ub - self.space.opt_lb) + self.space.opt_lb
        x       = samp[:, :self.space.num_numeric]
        xe      = samp[:, self.space.num_numeric:]
        for i, n in enumerate(self.space.numeric_names):
            if self.space.paras[n].is_discrete_after_transform:
                x[:, i] = x[:, i].round()
        df_samp = self.space.inverse_transform(x, xe)
        if fix_input is not None:
            for k, v in fix_input.items():
                df_samp[k] = v
        return df_samp

    @property
    def model_config(self):
        if self._model_config is None:
            if self.model_name == 'gp':
                cfg = {
                        'lr'           : 0.01,
                        'num_epochs'   : 100,
                        'verbose'      : False,
                        'noise_lb'     : 8e-4, 
                        'pred_likeli'  : False
                        }
            elif self.model_name == 'gpy':
                cfg = {
                        'verbose' : False,
                        'warp'    : True,
                        'space'   : self.space
                        }
            elif self.model_name == 'gpy_mlp':
                cfg = {
                        'verbose' : False
                        }
            elif self.model_name == 'rf':
                cfg =  {
                        'n_estimators' : 20
                        }
            else:
                cfg = {}
        else:
            cfg = deepcopy(self._model_config)

        if self.space.num_categorical > 0:
            cfg['num_uniqs'] = [len(self.space.paras[name].categories) for name in self.space.enum_names]
        return cfg

    def get_best_id(self, fix_input : dict = None) -> int:
        if fix_input is None:
            return np.argmin(self.y, axis=1)
        X = self.X.copy()
        y = self.y.copy()
        for k, v in fix_input.items():
            if X[k].dtype != 'float':
                crit = (X[k] != v).values
            else:
                crit = ((X[k] - v).abs() > np.finfo(float).eps).values
            y[crit]  = np.inf
        if np.isfinite(y).any():
            return np.argmin(y.reshape(-1))
        else:
            return np.argmin(self.y.reshape(-1))

    def suggest(self, n_suggestions=1, fix_input = None):
        if self.acq_cls != MACEEns and n_suggestions != 1:
            raise RuntimeError('Parallel optimization is supported only for MACE acquisition')
        if self.X[0].shape[0] < self.rand_sample:
            sample = self.quasi_sample(n_suggestions, fix_input)
            return sample
        else:
            Xs, Xes = [], []
            for i in range(self.num_ens):
                X, Xe = self.space.transform(self.X[i])
                Xs.append(X)
                Xes.append(Xe)
            ensemble = get_ensemble_models(self.model_name, self.num_ens, self.space.num_numeric, self.space.num_categorical, 1, **self.model_config)
            for i in range(self.num_ens):
                try:
                    if self.y[i].min() <= 0:
                        y = torch.FloatTensor(power_transform(self.y[i] / self.y[i].std(), method = 'yeo-johnson'))
                    else:
                        y = torch.FloatTensor(power_transform(self.y[i] / self.y[i].std(), method = 'box-cox'))
                        if y.std() < 0.5:
                            y = torch.FloatTensor(power_transform(self.y[i] / self.y[i].std(), method = 'yeo-johnson'))
                    if y.std() < 0.5:
                        raise RuntimeError('Power transformation failed')
                    ensemble[i].fit(Xs[i], Xes[i], y)

                except:
                    y     = torch.FloatTensor(self.y[i]).clone()
                    ensemble[i].fit(Xs[i], Xes[i], y)

            best_ids = self.get_best_id(fix_input)
            best_xs  = [x.iloc[i] for i, x in zip(best_ids, self.X)]
            #best_y  = y.min()

            py_best, ps2_best = ensemble[0].predict(*self.space.transform(best_xs[0]))
            if len(ensemble) != 1:
                for idx, mod in enumerate(ensemble[1:]):
                    py_mod_best, ps2_mod_best = mod.predict(*self.space.transform(best_xs[idx]))
                    py_best += py_mod_best
                    ps2_best += ps2_mod_best
            py_best /= len(ensemble)
            ps2_best /= len(ensemble)     
            py_best = py_best.detach().numpy().squeeze()
            ps_best = ps2_best.sqrt().detach().numpy().squeeze()

            iter  = max(1, self.X[0].shape[0] // n_suggestions)
            upsi  = 0.5
            delta = 0.01
            # kappa = np.sqrt(upsi * 2 * np.log(iter **  (2.0 + self.X.shape[1] / 2.0) * 3 * np.pi**2 / (3 * delta)))
            kappa = np.sqrt(upsi * 2 * ((2.0 + self.X[0].shape[1] / 2.0) * np.log(iter) + np.log(3 * np.pi**2 / (3 * delta))))

            #acq = self.acq_cls(model, best_y = py_best, kappa = kappa) # LCB < py_best
            acq = self.acq_cls(ensemble, best_y = py_best, kappa = kappa) # LCB < py_best
            mu  = MeanEns(ensemble)
            sig = SigmaEns(ensemble, linear_a = -1.)
            opt = EvolutionOpt(self.space, acq, pop = 100, iters = 100, verbose = False, es=self.es)
            rec = opt.optimize(initial_suggest = best_xs[0], fix_input = fix_input).drop_duplicates()
            unique_rec = self.check_unique(rec)
            unique_idx = [all(values) for values in zip(*unique_rec)]
            rec = rec[unique_idx]

            cnt = 0
            while rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(n_suggestions - rec.shape[0], fix_input)
                rand_rec = rand_rec[self.check_unique(rand_rec)]
                rec      = rec.append(rand_rec, ignore_index = True)
                cnt +=  1
                if cnt > 3:
                    # sometimes the design space is so small that duplicated sampling is unavoidable
                    break 
            if rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(n_suggestions - rec.shape[0], fix_input)
                rec      = rec.append(rand_rec, ignore_index = True)

            select_id = np.random.choice(rec.shape[0], n_suggestions, replace = False).tolist()
            x_guess   = []
            with torch.no_grad():
                py_all       = mu(*self.space.transform(rec)).squeeze().numpy()
                ps_all       = -1 * sig(*self.space.transform(rec)).squeeze().numpy()
                best_pred_id = np.argmin(py_all)
                best_unce_id = np.argmax(ps_all)
                if best_unce_id not in select_id and n_suggestions > 2:
                    select_id[0]= best_unce_id
                if best_pred_id not in select_id and n_suggestions > 2:
                    select_id[1]= best_pred_id
                rec_selected = rec.iloc[select_id].copy()
            return rec_selected

    def check_unique(self, rec : pd.DataFrame) -> [bool]:
        return [(~pd.concat([self.X[i], rec], axis = 0).duplicated().tail(rec.shape[0]).values).tolist() for i in range(self.num_ens)]

    def observe(self, X, y, rand_samps=False, ens_idx=0):
        """Feed an observation back.

        Parameters
        ----------
        X : pandas DataFrame
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,1)
            Corresponding values where objective has been evaluated
        y : int
            Which ensemble member to update the observation for (default 0)
        """
        #valid_id = np.where(np.isfinite(y.reshape(-1)))[0].tolist()
        XX = X.copy()
        yy = y.copy()
        
        #if X.shape[0] > 1:
        #    valid_id = [ens_idx]
        #    XX       = X.iloc[valid_id]
        #    yy       = y[[0]].reshape(-1, 1)
        #else:
        #    XX       = X.iloc[0]
        #    yy       = y[[0]].reshape(-1, 1)
        if rand_samps:
            self.X[ens_idx]   = self.X[ens_idx].append(XX, ignore_index = True)
            self.y[ens_idx]   = np.vstack([self.y[ens_idx], yy])
        
        else:
            for i in range(self.num_ens):
                self.X[i]  = self.X[i].append(XX, ignore_index = True)
                self.y[i]  = np.vstack([self.y[i], yy])
            
    @property
    def best_x(self)->pd.DataFrame:
        if self.X.shape[0] == 0:
            raise RuntimeError('No data has been observed!')
        else:
            return self.X.iloc[[self.y.argmin()]]

    @property
    def best_y(self)->float:
        if self.X.shape[0] == 0:
            raise RuntimeError('No data has been observed!')
        else:
            return self.y.min()
