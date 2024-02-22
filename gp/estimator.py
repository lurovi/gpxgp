from __future__ import annotations
import enum
from typing import Any
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import torch
from experiments.utils import make_fair_params
from gp.gp import LinearGP
from gp.gpgd.linear_gpgd import LinearGPGD
from gp.opgd.linear_opgd import LinearOPGD
from utils.evaluations import compute_linear_scaling
from utils.utils import set_seeds, make_fit_fun_gp, make_fit_fun_gpgd
from gp.prog_eval import ProgramEvaluator


class ParametrizedGPGD(BaseEstimator, RegressorMixin):
    def __init__(self,
                 mode: str = 'gp',
                 pop_size: int = 100,
                 dim_prg: int = 10,
                 max_dim_prg: int = 10,
                 enum_set: str = 'PLUS MINUS TIMES DIVIDE',
                 tournament_size: int = 4,
                 p_rand_op: float = 0.5,
                 rand_op_min: int = -5,
                 rand_op_max: int = 5,
                 p_m: float = 0.2,
                 min_con: int = -5,
                 max_con: int = 5,
                 linear_scaling: bool = False,
                 lr: float = 0.1,
                 comp_budget: int = 100,
                 e_in_evo: int = 0,
                 e_after_evo: int = 0,
                 seed: int | None = None,
                 verbose: bool = False,
                 parallel: bool = False
                 ) -> None:
        super().__init__()

        self.mode: str = mode
        self.pop_size: int = pop_size
        self.dim_prg: int = dim_prg
        self.max_dim_prg: int = max_dim_prg
        self.enum_set: str = enum_set
        self.tournament_size: int = tournament_size
        self.p_rand_op: float = p_rand_op
        self.rand_op_min: int = rand_op_min
        self.rand_op_max: int = rand_op_max
        self.p_m: float = p_m
        self.min_con: int = min_con
        self.max_con: int = max_con
        self.linear_scaling: bool = linear_scaling

        self.lr: float = lr
        self.comp_budget: int = comp_budget
        self.epochs_in_evolution: int = e_in_evo
        self.epochs_after_evolution: int = e_after_evo

        self.seed: int | None = seed
        self.verbose: bool = verbose
        self.parallel: bool = parallel


    def get_params(self, deep: bool = True) -> dict:
        d: dict = {}

        d['mode'] = self.mode
        d['pop_size'] = self.pop_size
        d['dim_prg'] = self.dim_prg
        d['max_dim_prg'] = self.max_dim_prg
        d['enum_set'] = self.enum_set
        d['tournament_size'] = self.tournament_size
        d['p_rand_op'] = self.p_rand_op
        d['rand_op_min'] = self.rand_op_min
        d['rand_op_max'] = self.rand_op_max
        d['p_m'] = self.p_m
        d['min_con'] = self.min_con
        d['max_con'] = self.max_con
        d['linear_scaling'] = self.linear_scaling
        d['lr'] = self.lr
        d['comp_budget'] = self.comp_budget
        d['epochs_in_evolution'] = self.epochs_in_evolution
        d['epochs_after_evolution'] = self.epochs_after_evolution
        d['seed'] = self.seed
        d['verbose'] = self.verbose
        d['parallel'] = self.parallel

        return d
    

    def set_params(self, **params) -> ParametrizedGPGD:
        for parameter, value in params.items():
            setattr(self, parameter, value)
        
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> ParametrizedGPGD:
        X, y = check_X_y(X, y, y_numeric=True)

        gp_params, in_evo_params, after_evo_params = make_fair_params(self.comp_budget, self.epochs_in_evolution, self.epochs_after_evolution)
        
        if self.mode == 'gp':
            alg = LinearGP
            lr = 0.0
            n_iter = gp_params['n_iter']
            epochs_in_evolution = gp_params['epochs_in_evolution']
            epochs_after_evolution = gp_params['epochs_after_evolution']
        elif self.mode == 'gpgda':
            alg = LinearGPGD
            lr = self.lr
            n_iter = in_evo_params['n_iter']
            epochs_in_evolution = in_evo_params['epochs_in_evolution']
            epochs_after_evolution = in_evo_params['epochs_after_evolution']
        elif self.mode == 'gpgdc':
            alg = LinearGPGD
            lr = self.lr
            n_iter = after_evo_params['n_iter']
            epochs_in_evolution = after_evo_params['epochs_in_evolution']
            epochs_after_evolution = after_evo_params['epochs_after_evolution']
        elif self.mode == 'opgda':
            alg = LinearOPGD
            lr = self.lr
            n_iter = in_evo_params['n_iter']
            epochs_in_evolution = in_evo_params['epochs_in_evolution']
            epochs_after_evolution = in_evo_params['epochs_after_evolution']
        elif self.mode == 'opgdc':
            alg = LinearOPGD
            lr = self.lr
            n_iter = after_evo_params['n_iter']
            epochs_in_evolution = after_evo_params['epochs_in_evolution']
            epochs_after_evolution = after_evo_params['epochs_after_evolution']
        else:
            raise ValueError(f'Invalid mode found ({self.mode}).')

        gp_params.clear()
        
        gp_params['n_iter'] = n_iter
        gp_params['epochs_in_evolution'] = epochs_in_evolution
        gp_params['epochs_after_evolution'] = epochs_after_evolution
        gp_params['lr'] = lr
        gp_params['pop_size'] = self.pop_size
        gp_params['dim_prg'] = self.dim_prg
        gp_params['max_dim_prg'] = self.max_dim_prg
        gp_params['tournament_size'] = self.tournament_size
        gp_params['p_rand_op'] = self.p_rand_op
        gp_params['rand_op_min'] = self.rand_op_min
        gp_params['rand_op_max'] = self.rand_op_max
        gp_params['p_m'] = self.p_m
        gp_params['min_con'] = self.min_con
        gp_params['max_con'] = self.max_con

        if self.seed is not None:
            set_seeds(self.seed)

        n_records: int = X.shape[0]
        self.n_features_in_: int = X.shape[1]
        self.opcodes_: enum.Enum = enum.Enum('opcodes', self.enum_set.strip())

        solver = alg(self.opcodes_, gp_params)

        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        x_train_list = [x_i for x_i in X.T]
        y.requires_grad_()

        if self.mode == 'gp':
            train_fit_fun = make_fit_fun_gp(x_train_list, y, self.opcodes_, reduction='mean')
        else:
            train_fit_fun = make_fit_fun_gpgd(x_train_list, y, reduction='mean')

        curr_best_prg, curr_best_val = solver.fit(train_fit_fun, verbose=self.verbose, parallel=self.parallel)
        
        self.best_prg_: Any = curr_best_prg
        self.best_val_: float = curr_best_val
        self.slope_: float = 1.0
        self.intercept_: float = 0.0

        if self.linear_scaling:
            p: np.ndarray = self.__apply_program(X, n_records)
            curr_slope, curr_intercept = compute_linear_scaling(y.detach().cpu().numpy(), p)
            self.slope_ = np.core.umath.clip(curr_slope, -1e+50, 1e+50)
            self.intercept_ = np.core.umath.clip(curr_intercept, -1e+50, 1e+50)

        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'The number of features of the fitted estimator is {self.n_features_in_}, but this dataset has {X.shape[1]} features instead.')

        n_records: int = X.shape[0]
        X = torch.from_numpy(X)
        
        return self.slope_ * self.__apply_program(X, n_records) + self.intercept_
        

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.predict(X)
    

    def __apply_program(self, X: torch.Tensor, n_records: int) -> np.ndarray:
        x_list = [x_i for x_i in X.T]
        with torch.no_grad():
            if self.mode == 'gp':
                pg_eval: ProgramEvaluator = ProgramEvaluator(self.opcodes_)
                p: np.ndarray = np.core.umath.clip(pg_eval(x_list, self.best_prg_).detach().cpu().numpy(), -1e+50, 1e+50)
            else:
                p: np.ndarray = np.core.umath.clip(self.best_prg_(x_list).detach().cpu().numpy(), -1e+50, 1e+50)

        if len(p) == 1:
            p = np.repeat(p[0], n_records)
        
        return p
