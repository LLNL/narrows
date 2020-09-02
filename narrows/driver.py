from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from pytz import timezone
from socket import gethostname
import time
import torch
import yaml

from .nn import ANNSlabSolver
from .mc1d import main as mc1dmain
from .sn1d import main as sn1dmain

from .writer import (
    initialize,
    write,
    close
)
from .version import (
    get_narrows_version,
    get_python_version,
    get_dependency_versions
)


def _predict_nn(nn_solver, z=None):
    start = time.time()
    flux = nn_solver.predict(z)
    prediction_runtime = time.time() - start
    return flux, prediction_runtime


def _train_nn(output_name, nn_solver):
    write('terse', 'Training neural network')
    start = time.time()
    train_result = nn_solver.train()
    training_runtime = time.time() - start

    return train_result, training_runtime


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
                    interval=deck.ctrl.interval,
                    gpu=deck.ctrl.gpu,
                    ahistory=deck.ctrl.ahistory,
                    hinterval=deck.ctrl.hinterval,
                    max_num_iter=deck.ctrl.max_num_iter)
    return nn_solver


def _run_nn(deck, mesh):
    nn_solver = _create_nn_object(deck, mesh)
    train_result, train_time = _train_nn(deck.ctrl.out, nn_solver)
    flux, pred_time = _predict_nn(nn_solver)

    if deck.ctrl.write_nn:
        with open(f'{deck.ctrl.out}.nn.pkl', 'wb') as f:
            pickle.dump(nn_solver, f)

    return flux, train_result, train_time, pred_time


def _run_mc(deck, mesh):
    write('terse', 'Running MC')
    start = time.time()
    tally = mc1dmain(mesh.edge.to_numpy(),
                     mesh.sigma_t.to_numpy(),
                     mesh.sigma_s0.to_numpy(),
                     mesh.sigma_s1.to_numpy(),
                     deck.src,
                     deck.ctrl.num_particles,
                     deck.ctrl.max_num_segments)
    runtime = time.time() - start
    return tally, runtime


def _run_sn(deck, mesh):
    write('terse', 'Running SN')
    start = time.time()
    result = sn1dmain(mesh.edge.to_numpy(),
                      mesh.sigma_t.to_numpy(),
                      mesh.sigma_s0.to_numpy(),
                      mesh.sigma_s1.to_numpy(),
                      mesh.source.to_numpy(),
                      deck.ctrl.ordinates,
                      deck.ctrl.sn_epsilon,
                      deck.ctrl.max_num_src_iter)
    runtime = time.time() - start
    return result, runtime


def _print_banner(argv):
    exe = argv[0]
    args = ' '.join(argv[1:])
    version = get_narrows_version()
    computer_name = gethostname()
    tz = timezone('US/Pacific')
    now = tz.localize(datetime.now())
    run_date = now.strftime('%d %B %Y (%A)')
    run_time = now.strftime('%H:%M:%S %z')

    write('terse', f'Executable       : {exe}')
    write('terse', f'Command line args: {args}')
    write('terse', f'Narrows version  : {version}')
    write('moderate', f'Computer name    : {computer_name}')
    write('moderate', f'Run date         : {run_date}')
    write('moderate', f'Run time         : {run_time}')
    write('terse', '')

    write('moderate', 'Copyright (c) 2020')
    write('moderate', 'Lawrence Livermore National Security, LLC')
    write('moderate', 'All Rights Reserved')
    write('moderate', '')

    python_version = get_python_version()
    write('moderate', f'Python version   : {python_version}')

    dependency_versions = get_dependency_versions()
    for dv in dependency_versions:
        dep, ver = dv
        lhs = f'{dep}'
        write('moderate', f'{lhs:17s}: {ver}')

    write('moderate', '')


def _write_input(yamlinput):
    write('none', yaml.dump(yamlinput), file_only=True)


def _set_seed(deck):
    np.random.seed(deck.ctrl.seed)
    torch.manual_seed(deck.ctrl.seed)


def _output_result(deck, mesh,
                   nn_flux, train_result, train_time, pred_time,
                   tally, mc_time,
                   sn_result, sn_time):

    number_of_algorithms = sum([deck.ctrl.nn, deck.ctrl.mc, deck.ctrl.sn])
    output = {'edge': mesh.edge}
    if number_of_algorithms == 1:
        if deck.ctrl.nn:
            output['algorithm'] = 'nn'
            output['flux'] = nn_flux
            output['loss'] = train_result['loss_history']
            output['spatial_loss'] = train_result['spatial_loss']
            if deck.ctrl.ahistory:
                output['flux_history'] = train_result['flux_history']
                output['spatial_loss_history'] = \
                    train_result['spatial_loss_history']
                output['hinterval'] = deck.ctrl.hinterval
            output['train_time'] = train_time
            output['pred_time'] = pred_time
        if deck.ctrl.mc:
            output['algorithm'] = 'mc'
            output['flux'] = tally.get_flux()
            output['time'] = mc_time
        if deck.ctrl.sn:
            output['algorithm'] = 'sn'
            output['flux'] = sn_result.I0
            output['time'] = sn_time
    elif number_of_algorithms > 1:
        if deck.ctrl.nn:
            output['nn_flux'] = nn_flux
            output['loss'] = train_result['loss_history']
            output['spatial_loss'] = train_result['spatial_loss']
            if deck.ctrl.ahistory:
                output['nn_flux_history'] = train_result['flux_history']
                output['spatial_loss_history'] = \
                    train_result['spatial_loss_history']
                output['hinterval'] = deck.ctrl.hinterval
            output['train_time'] = train_time
            output['pred_time'] = pred_time
        if deck.ctrl.mc:
            output['mc_flux'] = tally.get_flux()
            output['mc_time'] = mc_time
        if deck.ctrl.sn:
            output['sn_flux'] = sn_result.I0
            output['sn_time'] = sn_time
    else:
        write('terse', 'No flux calculated.')

    np.savez(f'{deck.ctrl.out}.npz', **output)

    write('moderate', get_runtimes(output))


def get_runtimes(output):
    if 'time' in output:
        runtime = output['time']
        return f'Runtime {runtime:.2e}'
    else:
        runtimes = {}
        runtime2column = {}

        if 'train_time' in output:
            runtimes['train_time'] = output['train_time']
            runtime2column['train_time'] = 'Train'
        if 'pred_time' in output:
            runtimes['pred_time'] = output['pred_time']
            runtime2column['pred_time'] = 'Predict'
        if 'mc_time' in output:
            runtimes['mc_time'] = output['mc_time']
            runtime2column['mc_time'] = 'MC'
        if 'sn_time' in output:
            runtimes['sn_time'] = output['sn_time']
            runtime2column['sn_time'] = 'SN'

        df = pd.DataFrame(runtimes, index=['Runtime'])
        df.rename(columns=runtime2column, inplace=True)
        pd.set_option('display.float_format', lambda x: '%.2e' % x)
        return df.transpose()


def run(deck, mesh):
    initialize(deck.ctrl.verb, deck.ctrl.out)
    _print_banner(deck.argv)
    _write_input(deck.yamlinput)
    _set_seed(deck)

    nn_flux = train_result = train_time = pred_time = None
    tally = mc_time = None
    sn_result = sn_time = None
    if deck.ctrl.nn:
        nn_flux, train_result, train_time, pred_time = _run_nn(deck, mesh)
    if deck.ctrl.mc:
        tally, mc_time = _run_mc(deck, mesh)
    if deck.ctrl.sn:
        sn_result, sn_time = _run_sn(deck, mesh)

    _output_result(deck, mesh,
                   nn_flux, train_result, train_time, pred_time,
                   tally, mc_time,
                   sn_result, sn_time)

    write('terse', '')
    write('terse', 'Narrows is finished.')
    close()

    return 0
