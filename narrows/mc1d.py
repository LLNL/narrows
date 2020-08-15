import numpy as np

from .writer import write

SMALL_DOUBLE = 1e-10
LARGE_DOUBLE = 1e50
SCATTER_REACTION = 0
ABSORB_REACTION = 1
FMT_STRING = '%20s %s@'


class Tally():
    def __init__(self, edges, num_particles):
        self._num_particles = float(num_particles)
        self._zone_lengths = edges[1:] - edges[:-1]
        self._num_zones = len(edges) - 1
        self._negcur = np.zeros(self._num_zones)
        self._negcur2 = np.zeros(self._num_zones)
        self._poscur = np.zeros(self._num_zones)
        self._poscur2 = np.zeros(self._num_zones)
        self._absorb = np.zeros(self._num_zones)
        self._left_escape_count = 0
        self._right_escape_count = 0

    def flux(self, dirs, weights, distances, particle_zone_idx):
        lengths = self._zone_lengths[particle_zone_idx]
        angular_flux = (distances * weights) / lengths

        neg_idx = np.where(dirs < 0)
        neg = True
        self._flux_helper(neg_idx, angular_flux, particle_zone_idx, neg)

        pos_idx = np.where(dirs > 0)
        neg = False
        self._flux_helper(pos_idx, angular_flux, particle_zone_idx, neg)

    def _flux_helper(self, idx, angular_flux, particle_zone_idx, neg):
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
                     ('flux', self.get_flux(),
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


class _Source():
    def __init__(self, source, num_particles):
        self.source = source
        self.num_particles = num_particles

    def sample_positions(self):
        length_weighted_mags = [src.magnitude * (src.end - src.start)
                                for src in self.source.values()]
        positions_per_src = []
        weights_per_src = []
        for lw_mag, src in zip(length_weighted_mags, self.source.values()):
            magnitude_ratio = lw_mag / sum(length_weighted_mags)
            num_src_particles = round(magnitude_ratio * self.num_particles)
            positions = _sample_uniform(src.start, src.end, num_src_particles)
            positions_per_src.append(positions)
            weights = np.full_like(positions, lw_mag / num_src_particles)
            weights_per_src.append(weights)

        positions = np.concatenate(positions_per_src)
        weights = np.concatenate(weights_per_src)
        return positions, weights

    def sample_direction_cosines(self):
        return _sample_uniform(-1, 1, self.num_particles)

    def __repr__(self):
        string = ('Source:@ num_particles {self.num_particles}@ '
                  'source {self.source}')

        return '\n'.join([x.strip() for x in string.split('@')])


class _Particles():
    def __init__(self, z, weight, mu, tally):
        self.z = z
        self.weight = weight
        self.mu = mu
        self.tally = tally
        self.segments = np.array([0] * len(z))
        self.alive = np.array([1] * len(z))

    def get_living_particle_indices(self):
        return self.alive.nonzero()[0]

    def compute_dist_to_boundary(self, particle_indices, edges):
        pos_dir_particle_indices = np.where(self.mu[particle_indices] > 0)
        pos_dir_edge_indices = \
            np.searchsorted(edges,
                            self.z[particle_indices][pos_dir_particle_indices])
        pos_dir_dist_to_boundary = (
            edges[pos_dir_edge_indices] -
            self.z[particle_indices][pos_dir_particle_indices])
        neg_dir_particle_indices = np.where(self.mu[particle_indices] < 0)
        neg_dir_edge_indices = (
           np.searchsorted(edges,
                           self.z[particle_indices][neg_dir_particle_indices])
           - 1)
        neg_dir_dist_to_boundary = (
            edges[neg_dir_edge_indices] -
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
        self.tally.flux(self.mu[particle_indices],
                        self.weight[particle_indices],
                        distance,
                        particle_zone_idx)
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
            action = 'ESCAPED'
            write('verbose', f'{action:>9s} {num_escaped}')

    def absorb(self, particle_indices, particle_zone_idx):
        self.tally.absorb(particle_zone_idx)
        action = 'ABSORBED'
        write('verbose', f'{action:>9s} {len(particle_indices)}')
        self.alive[particle_indices] = False

    def scatter(self, particle_indices, sigma_s0, sigma_s1, edges):
        # TODO TALLY SCATTERING REACTION
        action = 'SCATTERED'
        write('verbose', f'{action:>9s} {len(particle_indices)}')

        # linearly-isotropic scattering outgoing direction sampling
        # formula is from the last page of hmw4_Hint.pdf
        mubar = _safe_divide(sigma_s1, sigma_s0)
        mubar_at_position = mubar[np.searchsorted(edges,
                                                  self.z[particle_indices])]
        three_mu_times_mubar = (3 * self.mu[particle_indices] *
                                mubar_at_position)
        u_neg1_pos1 = 2 * _sample_uniform(size=len(particle_indices)) - 1
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


def _sample_dist_to_collision(positions, sigma_t, edges):
    dist = np.zeros(positions.shape)
    sigma_t_at_position = sigma_t[np.searchsorted(edges, positions)]

    idx_free_streaming = np.where(sigma_t_at_position == 0)
    dist[idx_free_streaming] = LARGE_DOUBLE

    idx_colliding = np.where(sigma_t_at_position != 0)
    len_idx_colliding = len(idx_colliding[0])
    dist[idx_colliding] = (-np.log(_sample_uniform(size=len_idx_colliding))
                           / sigma_t_at_position[idx_colliding])
    return dist


def _sample_collision(positions, scatter_prob, edges):
    scatter_prob_at_position = scatter_prob[np.searchsorted(edges, positions)]
    variates = _sample_uniform(size=len(positions))
    return variates > scatter_prob_at_position


def _sample_uniform(low=0.0, high=1.0, size=1):
    return np.random.uniform(low, high, size)


def _iteration_report(iteration_number, living_particle_indices, particles):
    write('verbose', f'--{iteration_number}--')
    write('verbose', particles)
    write('verbose', particles.tally)


def _safe_divide(a, b):
    assert a.shape == b.shape
    result = np.zeros(a.shape)
    idx = np.where(b != 0)
    result[idx] = a[idx] / b[idx]
    return result


def main(edges, sigma_t, sigma_s0, sigma_s1, source, num_particles,
         max_num_segments):

    scatter_prob = _safe_divide(sigma_s0, sigma_t)

    tally = Tally(edges, num_particles)
    src = _Source(source, num_particles)

    particles = _Particles(*src.sample_positions(),
                           src.sample_direction_cosines(),
                           tally)

    iteration_number = 0
    while np.any(particles.alive):
        living_particle_indices = particles.get_living_particle_indices()
        _iteration_report(iteration_number, living_particle_indices, particles)

        dist_to_collision = _sample_dist_to_collision(
                particles.z[living_particle_indices], sigma_t, edges)
        (dist_to_boundary, particle_zone_idx) = (
                particles.compute_dist_to_boundary(
                    living_particle_indices,
                    edges))

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
                                edges[-1])

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
        collisions = _sample_collision(particles.z[collision_particle_indices],
                                       scatter_prob, edges)
        particles.absorb(
                collision_particle_indices[
                    np.where(collisions == ABSORB_REACTION)],
                particle_zone_idx[dist_to_boundary > dist_to_collision][
                    np.where(collisions == ABSORB_REACTION)])
        particles.scatter(collision_particle_indices[
                          np.where(collisions == SCATTER_REACTION)],
                          sigma_s0, sigma_s1, edges)

        iteration_number += 1
        max_seg_indices = particles.segments > max_num_segments
        if np.any(max_seg_indices):
            write('terse', 'Warning: Killing %d particles which achieved max '
                  'num segments %d' % (len(max_seg_indices), max_num_segments))
            particles.alive[max_seg_indices] = False

    write('verbose', '--final state--')
    write('verbose', particles)

    return tally
