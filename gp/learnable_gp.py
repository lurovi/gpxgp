import itertools
import multiprocessing
from abc import ABC

import torch
from torch.optim import Adam

from gp.gp import LinearGP


class LearnableLinearGP(LinearGP, ABC):
    def __init__(self, opcodes, params=None):
        super().__init__(opcodes, params)

    def _fit_iteration(self, fit, pop, fitness, parallel):
        with torch.no_grad():
            pop = super()._fit_iteration(fit, pop, fitness, parallel)

        if parallel:
            pool = multiprocessing.Pool()

        params_list = [list(pop_i.parameters()) for pop_i in pop]
        optimizer = Adam(itertools.chain(*params_list), lr=self._params['lr'])
        for i in range(self._params['epochs_in_evolution']):
            optimizer.zero_grad()

            if parallel:
                vals = pool.map(fit, pop)
            else:
                vals = list(map(fit, pop))

            loss = sum(vals)
            loss.backward()
            optimizer.step()

        if parallel:
            pool.close()
            pool.join()

        return pop

    def fit(self, fit, verbose=False, parallel=True):
        best, best_value = super().fit(fit, verbose=verbose, parallel=parallel)

        optimizer = Adam(best.parameters(), lr=self._params['lr'])
        for i in range(self._params['epochs_after_evolution']):
            optimizer.zero_grad()

            loss = fit(best)
            loss.backward()
            optimizer.step()

            if verbose:
                print(f'Best fitness at gradint-descent iteration {i}: {loss.item()}')

        return best, fit(best)
