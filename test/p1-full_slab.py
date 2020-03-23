#!/usr/bin/env python
import driver

probname = 'p1-full_slab'
zstop = 1
argstr = (
 f'{probname} '
 f'--sigma_t 8 '
 f'--sigma_s0 0 '
 f'--sigma_s1 0 '
 f'--zstop {zstop} '
 f'--num_ordinates 4 '
 f'--num_hidden_layer_nodes 5 '
 f'--learning_rate 1e-3 '
 f'--epsilon_sn 1e-6 '
 f'--epsilon_nn 1e-13 '
 f'--num_sn_zones 50 '
 f'--num_mc_zones 50 '
 f'--num_nn_zones 50 '
 f'--num_particles 1000000 '
 f'--num_physical_particles 8 '
 f'--uniform_source_extent 0 {zstop} '
 f'--source_magnitude 8')

args = driver.parse_args(argstr.split())
driver.run(args)
