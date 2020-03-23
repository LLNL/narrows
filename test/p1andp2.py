#!/usr/bin/env python
import driver as d
argsp1 = d.parse_args('-q '
                      '-o datap1'
                      .split())
d.runp1orp2(argsp1, 'p1')

argsp2 = d.parse_args('-q '
                      '-o datap2 '
                      '--sigma_s1 5 '
                      '--uniform_source_extent 0 0 '  # NO uniform src Q(z)=1
                      '--point_source_location 0'
                      .split())
d.runp1orp2(argsp2, 'p2')
