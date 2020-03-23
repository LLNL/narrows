#!/usr/bin/env python

import numpy as np
import collections

SMALL_DOUBLE = 1e-10
LARGE_DOUBLE = 1e50
SCATTER_REACTION = 0
ABSORB_REACTION = 1
FMT_STRING = '%20s %s@'

__version__ = '0.0.1'


class Tally():
    def __init__(self, args, zone_edges):
        self._num_particles = float(args.num_particles)
        self._zone_lengths = zone_edges[1:] - zone_edges[:-1]
        self._num_zones = args.num_mc_zones
        self._negcur = np.zeros(self._num_zones)
        self._negcur2 = np.zeros(self._num_zones)
        self._poscur = np.zeros(self._num_zones)
        self._poscur2 = np.zeros(self._num_zones)
        self._absorb = np.zeros(self._num_zones)
        self._left_escape_count = 0
        self._right_escape_count = 0
        self._particle_weight = (args.num_physical_particles /
                                 args.num_particles)

    def flux(self, dirs, distances, particle_zone_idx):
        angular_flux = distances / self._zone_lengths[particle_zone_idx]

        neg_idx = np.where(dirs < 0)
        neg = True
        self.flux_helper(neg_idx, angular_flux, particle_zone_idx, neg)

        pos_idx = np.where(dirs > 0)
        neg = False
        self.flux_helper(pos_idx, angular_flux, particle_zone_idx, neg)

    def flux_helper(self, idx, angular_flux, particle_zone_idx, neg):
        bincount = np.bincount(particle_zone_idx[idx], angular_flux[idx])
        padded_bincount = np.pad(bincount,
                                 (0, self._num_zones - len(bincount)),
                                 'constant')
        if neg:
            self._negcur += padded_bincount
            self._negcur2 += padded_bincount * padded_bincount
        else:
            self._poscur += padded_bincount
            self._poscur2 += padded_bincount * padded_bincount

    def get_flux(self):
        return self._negcur + self._poscur

    def get_flux_squared(self):
        return self._negcur2 + self._poscur2

    def escape(self, num_alive_before_left_escape,
               num_alive_before_right_escape, num_alive_after):
        num_escaped_left = (num_alive_before_left_escape -
                            num_alive_before_right_escape)
        self._left_escape_count += num_escaped_left
        num_escaped_right = num_alive_before_right_escape - num_alive_after
        self._right_escape_count += num_escaped_right
        return num_escaped_left + num_escaped_right

    def absorb(self, particle_zone_idx):
        bincount = np.bincount(particle_zone_idx)
        padded_bincount = np.pad(bincount,
                                 (0, len(self._absorb) - len(bincount)),
                                 'constant')
        self._absorb += padded_bincount

    def __repr__(self):
        return self.dump(True, True)

    def dump(self, probabilities=True, tallies=False):
        pleftesc = self._left_escape_count / self._num_particles
        prightesc = self._right_escape_count / self._num_particles
        num_absorbed = np.sum(self._absorb)
        ratio_absorbed = num_absorbed / self._num_particles
        if probabilities and tallies:
            string = (FMT_STRING * 6) % \
                     ('flux', self.get_flux() * self._particle_weight,
                      'absorb', self._absorb,
                      'sum(absorb)', num_absorbed,
                      'sum(absorb) / total', ratio_absorbed,
                      'P(left esc)', pleftesc,
                      'P(right esc)', prightesc)
        elif probabilities:
            string = (FMT_STRING * 4) % \
                     ('sum(absorb)', num_absorbed,
                      'sum(absorb) / total', ratio_absorbed,
                      'P(left esc)', pleftesc,
                      'P(right esc)', prightesc)
        elif tallies:
            string = (FMT_STRING * 2) % \
                     ('flux', self.get_flux(),
                      'absorb', self._absorb)
        else:
            string = 'Must ask for probabilities or tallies (or both).'

        return '\n'.join(string.split('@'))


class Source():
    def __init__(self, args):
        self.num_particles = args.num_particles
        if args.point_source_location is not None:
            self.position = args.point_source_location + SMALL_DOUBLE
            self.point_source = True
        else:
            self.position = args.uniform_source_extent
            self.point_source = False

    def sample_positions(self):
        if self.point_source:
            return np.array([self.position] * self.num_particles)
        else:
            low = self.position[0]
            high = self.position[1]
            return sample_uniform(low, high, self.num_particles)

    def sample_direction_cosines(self):
        return sample_uniform(-1, 1, self.num_particles)

    def __repr__(self):
        string = 'Source:@ num_particles %d@ position %s@ point_source %s' % \
                (self.num_particles, str(self.position),
                 str(self.point_source))
        return '\n'.join([x.strip() for x in string.split('@')])


class Particles():
    def __init__(self, z, mu, tally, verbose):
        self.z = z
        self.mu = mu
        self.tally = tally
        self.verbose = verbose
        self.segments = np.array([0] * len(z))
        self.alive = np.array([1] * len(z))

    def get_living_particle_indices(self):
        return self.alive.nonzero()[0]

    def compute_dist_to_boundary(self, particle_indices, zone_edges):
        pos_dir_particle_indices = np.where(self.mu[particle_indices] > 0)
        pos_dir_edge_indices = \
            np.searchsorted(zone_edges,
                            self.z[particle_indices][pos_dir_particle_indices])
        pos_dir_dist_to_boundary = (
            zone_edges[pos_dir_edge_indices] -
            self.z[particle_indices][pos_dir_particle_indices])
        neg_dir_particle_indices = np.where(self.mu[particle_indices] < 0)
        neg_dir_edge_indices = (
           np.searchsorted(zone_edges,
                           self.z[particle_indices][neg_dir_particle_indices])
           - 1)
        neg_dir_dist_to_boundary = (
            zone_edges[neg_dir_edge_indices] -
            self.z[particle_indices][neg_dir_particle_indices])

        dist_to_boundary = np.zeros(len(particle_indices))
        dist_to_boundary[pos_dir_particle_indices] = \
            pos_dir_dist_to_boundary / \
            self.mu[particle_indices][pos_dir_particle_indices]
        dist_to_boundary[neg_dir_particle_indices] = \
            neg_dir_dist_to_boundary / \
            self.mu[particle_indices][neg_dir_particle_indices]

        # The index of the zone that the particle is in (before it moves)
        # For the mesh
        # 0 1 2 3 4
        # the index 0 means the particle is somewhere between 0 and 1
        # This information is used to accumulate zonal tallies (e.g. flux)
        particle_zone_idx = np.zeros(len(particle_indices), dtype=np.int)
        particle_zone_idx[pos_dir_particle_indices] = pos_dir_edge_indices - 1
        particle_zone_idx[neg_dir_particle_indices] = neg_dir_edge_indices

        return dist_to_boundary, particle_zone_idx

    def move_particle(self, particle_indices, distance, particle_zone_idx,
                      zstop=LARGE_DOUBLE):
        self.tally.flux(self.mu[particle_indices], distance, particle_zone_idx)
        self.segments[particle_indices] += 1
        horizontal_distance = ((distance + SMALL_DOUBLE) *
                               self.mu[particle_indices])
        self.z[particle_indices] += horizontal_distance
        if zstop != LARGE_DOUBLE:
            # This is not a collision, so check for escapees
            num_alive_before_left_escape = np.sum(self.alive)
            # escape to the left
            # self.alive[particle_indices][np.where(self.z[particle_indices] <
            #                                       0)] = False
            # Why does the above line not work?
            self.alive[self.z < 0] = False
            # escape to the right
            # self.alive[particle_indices][np.where(self.z[particle_indices] >
            #                                       zstop)] = False
            # Why does the above line not work?
            num_alive_before_right_escape = np.sum(self.alive)
            self.alive[self.z > zstop] = False
            num_escaped = self.tally.escape(num_alive_before_left_escape,
                                            num_alive_before_right_escape,
                                            np.sum(self.alive))
            if self.verbose:
                print('%9s %d' % ('ESCAPED', num_escaped))

    def absorb(self, particle_indices, particle_zone_idx):
        self.tally.absorb(particle_zone_idx)
        if self.verbose:
            print('%9s %d' % ('ABSORBED', len(particle_indices)))
        self.alive[particle_indices] = False

    def scatter(self, particle_indices, sigma_s0, sigma_s1):
        # TODO TALLY SCATTERING REACTION
        if self.verbose:
            print('%9s %d' % ('SCATTERED', len(particle_indices)))

        # linearly-isotropic scattering outgoing direction sampling
        # formula is from the last page of hmw4_Hint.pdf
        mubar = safe_divide(sigma_s1, sigma_s0)
        three_mu_times_mubar = 3 * self.mu[particle_indices] * mubar
        u_neg1_pos1 = 2 * sample_uniform(size=len(particle_indices)) - 1
        self.mu[particle_indices] = ((2 * u_neg1_pos1 + three_mu_times_mubar) /
                                     (1 + np.sqrt(1 + three_mu_times_mubar *
                                                  (2 * u_neg1_pos1 +
                                                   three_mu_times_mubar))))

    def __repr__(self):
        return self.dump(np.array([True] * len(self.z)))

    def dump(self, indices):
        string = (FMT_STRING * 4) % \
                 ('z', str(self.z),
                  'mu', str(self.mu),
                  'segments', str(self.segments),
                  'alive', str(self.alive))
        return '\n'.join(string.split('@'))


def sample_dist_to_collision(sample_size, sigma_t):
    if sigma_t == 0:
        return np.array([LARGE_DOUBLE] * sample_size)
    else:
        return -np.log(sample_uniform(size=sample_size)) / sigma_t


def sample_collision(sample_size, reaction_cdf):
    return np.searchsorted(reaction_cdf, sample_uniform(size=sample_size))


def sample_uniform(low=0.0, high=1.0, size=1):
    return np.random.uniform(low, high, size)


def print_iteration_report(iteration_number, living_particle_indices,
                           particles, verbose):
    if verbose:
        print('--%d--' % iteration_number)
        print(particles)
        print(particles.tally)


def safe_divide(a, b):
    return 0 if b == 0 else a / b


def soft_zero(value):
    if abs(value) < (SMALL_DOUBLE * 2):
        return True
    else:
        return False


def main(args):

    # Set random number seed
    np.random.seed(args.seed)

    # J
    zone_edges = np.linspace(0, args.zstop, args.num_mc_zones + 1)
    if args.verbose:
        print('zone_edges', zone_edges)

    scatter_prob = safe_divide(args.sigma_s0, args.sigma_s0 + args.sigma_t)
    reaction_cdf = [scatter_prob, 1]

    tally = Tally(args, zone_edges)
    src = Source(args)
    particles = Particles(src.sample_positions(),
                          src.sample_direction_cosines(), tally, args.verbose)

    iteration_number = 0
    while np.any(particles.alive):
        living_particle_indices = particles.get_living_particle_indices()
        print_iteration_report(iteration_number, living_particle_indices,
                               particles, args.verbose)

        dist_to_collision = sample_dist_to_collision(
                len(living_particle_indices), args.sigma_t)
        (dist_to_boundary, particle_zone_idx) = (
                particles.compute_dist_to_boundary(
                    living_particle_indices,
                    zone_edges))

        # Handle boundary crossing
        boundary_crossing_particle_indices = living_particle_indices[
                dist_to_boundary < dist_to_collision]
        boundary_crossing_particle_distances = dist_to_boundary[
                dist_to_boundary < dist_to_collision]
        boundary_crossing_particle_zone_idx = particle_zone_idx[
                dist_to_boundary < dist_to_collision]
        particles.move_particle(boundary_crossing_particle_indices,
                                boundary_crossing_particle_distances,
                                boundary_crossing_particle_zone_idx,
                                args.zstop)

        # Handle collision
        collision_particle_indices = living_particle_indices[dist_to_boundary >
                                                             dist_to_collision]
        collision_particle_distances = dist_to_collision[dist_to_boundary >
                                                         dist_to_collision]
        collision_particle_zone_idx = particle_zone_idx[dist_to_boundary >
                                                        dist_to_collision]
        particles.move_particle(collision_particle_indices,
                                collision_particle_distances,
                                collision_particle_zone_idx)
        collisions = sample_collision(len(collision_particle_indices),
                                      reaction_cdf)
        particles.absorb(
                collision_particle_indices[
                    np.where(collisions == ABSORB_REACTION)],
                particle_zone_idx[dist_to_boundary > dist_to_collision][
                    np.where(collisions == ABSORB_REACTION)])
        particles.scatter(collision_particle_indices[
                          np.where(collisions == SCATTER_REACTION)],
                          args.sigma_s0, args.sigma_s1)

        iteration_number += 1
        max_seg_indices = particles.segments > args.max_num_segments
        if np.any(max_seg_indices):
            print('Warning: Killing %d particles which achieved max num '
                  'segments %d' % (len(max_seg_indices),
                                   args.max_num_segments))
            particles.alive[max_seg_indices] = False

    if args.verbose:
        print('--final state--')
        print(particles)

    Result = collections.namedtuple('Result',
                                    'zone_edges tally')
    return Result(zone_edges, tally)

    return particles.tally
