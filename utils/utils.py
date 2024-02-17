import random
import enum

import numpy as np
import torch
import yaml
from functools import partial
from gp.prog_eval import ProgramEvaluator


def read_yaml_hyperparams(filename: str):
    with open(filename, 'r') as f:
        hyperparams = yaml.safe_load(f)

    return hyperparams


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def quadratic_seed(base_seed: int, offset_seed: int, multiply_by_prime_number: bool = True) -> int:
    return base_seed + (offset_seed * offset_seed) * (31 if multiply_by_prime_number else 1)


def cubic_seed(base_seed: int, offset_seed: int, multiply_by_prime_number: bool = True) -> int:
    return base_seed + (offset_seed * offset_seed * offset_seed) * (31 if multiply_by_prime_number else 1)


def make_fit_fun_gp(x, y, opcodes, reduction='mean'):
    pg_eval: ProgramEvaluator = ProgramEvaluator(opcodes)
    if reduction == 'mean':
        return partial(fit_fun_gp_mean, x=x, y=y, pg_eval=pg_eval)
    if reduction == 'sum':
        return partial(fit_fun_gp_sum, x=x, y=y, pg_eval=pg_eval)


def make_fit_fun_gpgd(x, y, reduction='mean'):
    """
    fitness function for gpgd routine
    """
    if reduction == 'mean':
        return partial(fit_fun_gpgd_mean, x=x, y=y)
    elif reduction == 'sum':
        return partial(fit_fun_gpgd_sum, x=x, y=y)
    else:
        return partial(fit_fun_gpgd_mean, x=x, y=y)


def fit_fun_gp_mean(prg, x, y, pg_eval):
    return torch.mean((y - pg_eval(x, prg)) ** 2)


def fit_fun_gp_sum(prg, x, y, pg_eval):
    return torch.sum((y - pg_eval(x, prg)) ** 2)


def fit_fun_gpgd_mean(prg, x, y):
    return torch.mean((y - prg(x)) ** 2)


def fit_fun_gpgd_sum(prg, x, y):
    return torch.sum((y - prg(x)) ** 2)

