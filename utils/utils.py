import random
import enum

import numpy as np
import torch
import yaml

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
        return lambda prg: torch.mean((y - pg_eval(x, prg)) ** 2)
    if reduction == 'sum':
        return lambda prg: torch.sum((y - pg_eval(x, prg)) ** 2)


def make_fit_fun_gpgd(x, y, reduction='mean'):
    """
    fitness function for gpgd routine
    """
    if reduction == 'mean':
        return lambda prg: torch.mean((y - prg(x)) ** 2)
    elif reduction == 'sum':
        return lambda prg: torch.sum((y - prg(x)) ** 2)
    else:
        return lambda prg: torch.mean((y - prg(x)) ** 2)

