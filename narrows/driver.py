import numpy as np
import pickle
import time
import torch

from .nn import ANNSlabSolver
from .mc1d import main as mc1dmain
from .sn1d import main as sn1dmain


def _predict_nn(nn_solver, z=None):
    start = time.time()
    flux = nn_solver.predict(z)
    prediction_runtime = time.time() - start
    return flux, prediction_runtime


def _train_nn(output_name, nn_solver):
    print("Training neural network")
    start = time.time()
    loss_history = nn_solver.train()
    training_runtime = time.time() - start

    with open(f'{output_name}.loss.pkl', 'wb') as f:
        pickle.dump(loss_history, f)
    return training_runtime


def _create_nn_object(deck, mesh):
    nn_solver = ANNSlabSolver(deck.ctrl.ordinates,
                              deck.ctrl.hidden_layer_nodes,
                              mesh.edge.to_numpy(),
                              mesh.sigma_t.to_numpy(),
                              mesh.sigma_s0.to_numpy(),
                              mesh.sigma_s1.to_numpy(),
                              mesh.source.to_numpy(),
                              eps=deck.ctrl.epsilon,
                              tensorboard=deck.ctrl.tensorboard)
    return nn_solver


def _run_nn(deck, mesh):
    nn_solver = _create_nn_object(deck, mesh)
    train_time = _train_nn(deck.ctrl.out, nn_solver)
    flux, pred_time = _predict_nn(nn_solver)

    with open(f'{deck.ctrl.out}.nn.pkl', 'wb') as f:
        pickle.dump(nn_solver, f)

    return flux, train_time, pred_time


def _run_mc(deck, mesh):
    start = time.time()
    tally = mc1dmain(mesh.edge.to_numpy(),
                     mesh.sigma_t.to_numpy(),
                     mesh.sigma_s0.to_numpy(),
                     mesh.sigma_s1.to_numpy(),
                     deck.src,
                     deck.ctrl.num_particles,
                     deck.ctrl.num_physical_particles,
                     deck.ctrl.max_num_segments,
                     deck.ctrl.verbose)
    runtime = time.time() - start
    return tally, runtime


def _run_sn(deck, mesh):
    start = time.time()
    result = sn1dmain(mesh.edge.to_numpy(),
                      mesh.sigma_t.to_numpy(),
                      mesh.sigma_s0.to_numpy(),
                      mesh.sigma_s1.to_numpy(),
                      mesh.source.to_numpy(),
                      deck.ctrl.ordinates,
                      deck.ctrl.sn_epsilon,
                      deck.ctrl.max_num_iter,
                      deck.ctrl.verbose)
    runtime = time.time() - start
    return result, runtime


def _print_banner(deck):
    # TODO print copyright, version, etc.
    pass


def _set_seed(deck):
    np.random.seed(deck.ctrl.seed)
    torch.manual_seed(deck.ctrl.seed)


def _write_result(deck, mesh, nn_flux, train_time, pred_time, tally, mc_time,
                  sn_result, sn_time):
    number_of_algorithms = sum([deck.ctrl.nn, deck.ctrl.mc, deck.ctrl.sn])
    output = {'edge': mesh.edge}
    runtimes = {}
    if number_of_algorithms == 1:
        if deck.ctrl.nn:
            output['flux'] = nn_flux
            runtimes['train_time'] = train_time
            runtimes['pred_time'] = pred_time
        if deck.ctrl.mc:
            output['flux'] = tally.get_flux()
            runtimes['time'] = mc_time
        if deck.ctrl.sn:
            output['flux'] = sn_result.I0
            runtimes['time'] = sn_time
    elif number_of_algorithms > 1:
        if deck.ctrl.nn:
            output['nn_flux'] = nn_flux
            runtimes['train_time'] = train_time
            runtimes['pred_time'] = pred_time
        if deck.ctrl.mc:
            output['mc_flux'] = tally.get_flux()
            runtimes['mc_time'] = mc_time
        if deck.ctrl.sn:
            output['sn_flux'] = sn_result.I0
            runtimes['sn_time'] = sn_time
    else:
        print('No flux calculated.')

    np.savez(f'{deck.ctrl.out}.npz', **output)

    with open(f'{deck.ctrl.out}.time', 'w') as f:
        f.write('%s\n' % str(runtimes))


def run(deck, mesh):
    _print_banner(deck)
    _set_seed(deck)

    nn_flux = train_time = pred_time = tally = mc_time = sn_result = sn_time \
            = None
    if deck.ctrl.nn:
        nn_flux, train_time, pred_time = _run_nn(deck, mesh)
    if deck.ctrl.mc:
        tally, mc_time = _run_mc(deck, mesh)
    if deck.ctrl.sn:
        sn_result, sn_time = _run_sn(deck, mesh)

    _write_result(deck, mesh, nn_flux, train_time, pred_time, tally, mc_time,
                  sn_result, sn_time)
    return 0
