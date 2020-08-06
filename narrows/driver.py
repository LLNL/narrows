from datetime import datetime
import numpy as np
import pickle
from pytz import timezone
from socket import gethostname
import time
import torch
import yaml

from .nn import ANNSlabSolver
from .mc1d import main as mc1dmain
from .sn1d import main as sn1dmain

from .writer import write
from .version import get_version


def _predict_nn(nn_solver, z=None):
    start = time.time()
    flux = nn_solver.predict(z)
    prediction_runtime = time.time() - start
    return flux, prediction_runtime


def _train_nn(output_name, nn_solver):
    write('moderate', 'Training neural network')
    start = time.time()
    loss_history = nn_solver.train()
    training_runtime = time.time() - start

    with open(f'{output_name}.loss.pkl', 'wb') as f:
        pickle.dump(loss_history, f)
    return training_runtime


def _create_nn_object(deck, mesh):
    nn_solver = ANNSlabSolver(
                    deck.ctrl.ordinates,
                    deck.ctrl.hidden_layer_nodes,
                    mesh.edge.to_numpy(),
                    mesh.sigma_t.to_numpy(),
                    mesh.sigma_s0.to_numpy(),
                    mesh.sigma_s1.to_numpy(),
                    mesh.source.to_numpy(),
                    gamma_l=deck.ctrl.gamma_l,
                    gamma_r=deck.ctrl.gamma_r,
                    eps=deck.ctrl.epsilon,
                    tensorboard=deck.ctrl.tensorboard,
                    interval=deck.ctrl.interval)
    return nn_solver


def _run_nn(deck, mesh):
    nn_solver = _create_nn_object(deck, mesh)
    train_time = _train_nn(deck.ctrl.out, nn_solver)
    flux, pred_time = _predict_nn(nn_solver)

    with open(f'{deck.ctrl.out}.nn.pkl', 'wb') as f:
        pickle.dump(nn_solver, f)

    return flux, train_time, pred_time


def _run_mc(deck, mesh):
    write('moderate', 'Running MC')
    start = time.time()
    tally = mc1dmain(mesh.edge.to_numpy(),
                     mesh.sigma_t.to_numpy(),
                     mesh.sigma_s0.to_numpy(),
                     mesh.sigma_s1.to_numpy(),
                     deck.src,
                     deck.ctrl.num_particles,
                     deck.ctrl.num_physical_particles,
                     deck.ctrl.max_num_segments)
    runtime = time.time() - start
    return tally, runtime


def _run_sn(deck, mesh):
    write('moderate', 'Running SN')
    start = time.time()
    result = sn1dmain(mesh.edge.to_numpy(),
                      mesh.sigma_t.to_numpy(),
                      mesh.sigma_s0.to_numpy(),
                      mesh.sigma_s1.to_numpy(),
                      mesh.source.to_numpy(),
                      deck.ctrl.ordinates,
                      deck.ctrl.sn_epsilon,
                      deck.ctrl.max_num_iter)
    runtime = time.time() - start
    return result, runtime


def _print_banner(argv):
    exe = argv[0]
    args = ' '.join(argv[1:])
    version = get_version()
    computer_name = gethostname()
    tz = timezone('US/Pacific')
    now = tz.localize(datetime.now())
    run_date = now.strftime('%d %B %Y (%A)')
    run_time = now.strftime('%H:%M:%S %z')
    write('moderate', f'Executable       : {exe}')
    write('moderate', f'Command line args: {args}')
    write('moderate', f'Narrows version  : {version}')
    write('moderate', f'Computer name    : {computer_name}')
    write('moderate', f'Run date         : {run_date}')
    write('moderate', f'Run time         : {run_time}')
    write('moderate', '')
    write('verbose', 'Copyright (c) 2020')
    write('verbose', 'Lawrence Livermore National Security, LLC')
    write('verbose', 'All Rights Reserved')
    write('verbose', '')


def _write_input(yamlinput):
    write('none', yaml.dump(yamlinput), file_only=True)


def _set_seed(deck):
    np.random.seed(deck.ctrl.seed)
    torch.manual_seed(deck.ctrl.seed)


def _output_result(deck, mesh, nn_flux, train_time, pred_time, tally, mc_time,
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
        write('terse', 'No flux calculated.')

    np.savez(f'{deck.ctrl.out}.npz', **output)

    with open(f'{deck.ctrl.out}.time', 'w') as f:
        f.write('%s\n' % str(runtimes))


def run(deck, mesh):
    _print_banner(deck.argv)
    _write_input(deck.yamlinput)
    _set_seed(deck)

    nn_flux = train_time = pred_time = tally = mc_time = sn_result = sn_time \
            = None
    if deck.ctrl.nn:
        nn_flux, train_time, pred_time = _run_nn(deck, mesh)
    if deck.ctrl.mc:
        tally, mc_time = _run_mc(deck, mesh)
    if deck.ctrl.sn:
        sn_result, sn_time = _run_sn(deck, mesh)

    _output_result(deck, mesh, nn_flux, train_time, pred_time, tally, mc_time,
                   sn_result, sn_time)
    return 0
