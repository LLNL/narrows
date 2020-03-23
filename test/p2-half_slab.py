#!/usr/bin/env python
import driver

probname = 'p2-half_slab'
argstr = (
 f'{probname} '
 f'--sigma_t 8 '
 f'--sigma_s0 6.4 '
 f'--sigma_s1 1.6 '
 f'--zstop 1 '
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
 f'--uniform_source_extent 0 0.5 '
 f'--source_magnitude 8')

args = driver.parse_args(argstr.split())
driver.run(args)
