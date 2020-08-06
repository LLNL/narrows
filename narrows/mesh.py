import numpy as np
import pandas as pd


def create_mesh(deck):
    edges = [np.linspace(x.start, x.end, deck.ctrl.cells_per_region + 1,
                         dtype=np.float32) for x in deck.reg.values()]
    edges = np.unique(np.concatenate(edges))
    mats = [None] * len(edges)
    sigma_a = np.zeros(len(edges), dtype=np.float32)
    sigma_s0 = np.zeros(len(edges), dtype=np.float32)
    sigma_s1 = np.zeros(len(edges), dtype=np.float32)
    for edge_index, edge in enumerate(edges):
        for reg in deck.reg.values():
            if (reg.start <= edge) and (edge <= reg.end):
                mats[edge_index] = reg.mat
                sigma_a[edge_index] = deck.mat[reg.mat].sigma_a
                sigma_s0[edge_index] = deck.mat[reg.mat].sigma_s0
                sigma_s1[edge_index] = deck.mat[reg.mat].sigma_s1
                break

    source = np.zeros(len(edges), dtype=np.float32)
    for edge_index, edge in enumerate(edges):
        for src in deck.src.values():
            if (src.start <= edge) and (edge <= src.end):
                source[edge_index] = src.magnitude
                break

    sigma_t = sigma_a + sigma_s0
    df = pd.DataFrame({'edge': edges,
                       'mat': mats,
                       'sigma_t': sigma_t,
                       'sigma_a': sigma_a,
                       'sigma_s0': sigma_s0,
                       'sigma_s1': sigma_s1,
                       'source': source})
    return df
