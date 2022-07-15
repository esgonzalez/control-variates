#!/usr/bin/env python3
"""Prototype using control variates for variance reduction."""
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import collections as cl

class Surface:
    def __init__(self, name, bc):
        self.name = name
        self.bc = bc
        self.estimators = []

    def add_estimator(self, est):
        self.estimators.append(est)

    def evaluate(self, axis, position):
        raise NotImplementedError()

    def __pos__(self):
        return (self, +1)

    def __neg__(self):
        return (self, -1)

class Plane(Surface):
    def __init__(self, name, a, b, c, d, bc=None):
        Surface.__init__(self, name, bc)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def distance(self, pos, dof):
        dist = (self.d - self.a * pos[0] - self.b * pos[1] - self.c * pos[2]) / (self.a * dof[0] + self.b * dof[1] + self.c * dof[2])

        if (dist > 0.0):
            return dist

        return np.inf

    def evaluate(self, pos):
        return self.a*pos[0] + self.b*pos[1] + self.c*pos[2] - self.d

class Sphere(Surface):
    def __init__(self, name, center, radius, bc=None):
        Surface.__init__(self, name, bc)
        self.center = center
        self.radius = radius

    def distance(self, pos, dof):
        """Distance between particle and origin-centered sphere."""
        # Compute discriminant
        dir_dot_pos = np.dot(dof, pos)
        discriminant = np.square(dir_dot_pos) - np.dot(pos, pos) + np.square(self.radius)
        if discriminant < 0.:
            # line and sphere do not intersect
            return np.inf
        if discriminant > 0.:
            # two solutions exist, return smallest positive
            value = -dir_dot_pos + np.sqrt(discriminant)
            return value if value > 0 else np.inf
        # line is tangent to sphere
        raise RuntimeError

    def evaluate(self, pos):
        return np.sum(np.square(pos - self.center)) - self.radius**2

class Material():
    def __init__(self, xs):
        self.xs = xs
        self.rxn_probs = {}
        self.xs_totals = {}
        self.compute_totals()

    def compute_totals(self):
        for g in self.xs:
            scatter_xs = sum(self.xs[g]['scatter'].values())
            capture_xs = self.xs[g]['capture']
            self.rxn_probs[g] = {'scatter': scatter_xs, 'capture': capture_xs}
            self.xs_totals[g] = scatter_xs + capture_xs


class Cell():
    def __init__(self, name, mat):
        self.name = name
        self.mat = mat
        self.surfaces = []
        self.senses = []
        self.estimators = []

    def add_estimator(self, est):
        self.estimators.append(est)

    def add_surface(self, surface_and_sense):
        surface, int_sense = surface_and_sense
        self.surfaces.append(surface)
        self.senses.append(self._convert_sense(int_sense))

    def add_surfaces(self, surfaces_and_senses):
        for surface_and_sense in surfaces_and_senses:
            self.add_surface(surface_and_sense)

    def _convert_sense(self, int_sense):
        return False if int_sense == -1 else True

    def get_closest_surface(self, pos, dof):
        distances = {surface: surface.distance(pos, dof) for surface in self.surfaces}
        closest = min(distances, key=distances.get)
        return closest, distances[closest]

    def inside(self, position):
        for surface, sense in zip(self.surfaces, self.senses):
            evaluation = surface.evaluate(position)
            if not np.isclose(evaluation, 0.0, atol=1e-08):
                point_sense = True if evaluation > 0.0 else False
                if (point_sense != sense):
                    return False
        return True

class Geometry():
    def __init__(self, cells, surfs):
        self.cells = cells
        self.surfs = surfs

    def find_cell(self, pos):
        for cell in self.cells.values():
            if cell.inside(pos):
                return cell
        raise RuntimeError


class Particle():
    def __init__(self, history, pos, group, cell):
        self.history = history
        self.pos = pos
        self.dof = self.sample_isotropic()
        self.group = group
        self.cell = cell
        self.alive = True

    def sample_isotropic(self):
        """
        Sample a point on the unit sphere istotropically.
        """
        costheta = 2. * np.random.random() - 1. # mu
        sintheta = np.sqrt(1. - costheta * costheta)
        phi = 2. * np.pi * np.random.random()
        cosphi = np.cos(phi)
        sinphi = np.sqrt(1. - cosphi * cosphi)
        return np.array([costheta, sintheta * cosphi, sintheta * sinphi])

class Estimator():
    def __init__(self, name, groups):
        self.name = name
        self.history_sum = {g: 0.0 for g in groups}
        self.history_sum_sq = {g: 0.0 for g in groups}
        self.covariance_estimators = {}
        self.covariance_moment = {}

    def add_covariance(self, est):
        self.covariance_estimators[est.name] = est
        self.covariance_moment[est.name] = {g: 0.0 for g in est.history_sum.keys()}

    def score(self, group, value):
        self.history_sum[group] += value

    def end_history(self):
        for group in self.history_sum.keys():
            self.history_sum_sq[group] += self.history_sum[group] ** 2
            for cov_est in self.covariance_estimators.keys():
                self.covariance_moment[cov_est][group] += self.covariance_estimators[cov_est].history_sum[group] * self.history_sum[group]

    def begin_history(self):
        self.history_sum = {g: 0.0 for g in self.history_sum.keys()}
        self.history_sum_sq = {g: 0.0 for g in self.history_sum.keys()}
        self.covariance_moment = {est: {g: 0.0 for g in self.history_sum.keys()} for est in self.covariance_moment.keys()}

def simulate_particle(history):
    """Simulate the particle history."""

    estimators = {}

    # Set up estimators
    for cell in geometry.cells.values():
        for est in cell.estimators:
            estimators[est] = Estimator(est, groups)

    for surf in geometry.surfs.values():
        for est in surf.estimators:
            estimators[est] = Estimator(est, groups)

    estimators['x_zero_current'].add_covariance(estimators['box1_capture'])

    # Run simulation
    np.random.seed(history)

    p = Particle(history, np.array([-0.5, 0.0, 0.0]), 1, geometry.cells['box1'])

    while p.alive:
        # Sample next event
        total_xs = p.cell.mat.xs_totals[p.group]

        closest_surface, surf_dist = p.cell.get_closest_surface(p.pos, p.dof)

        distances = {
            'collision': (- np.log(np.random.random()) / total_xs),
            'surface_crossing': surf_dist}

        min_event = min(distances, key=distances.get)
        distance = distances[min_event]
        p.pos += p.dof * (distance + 1e-16)

        # Update particle to new state
        if min_event == 'collision':
            # Sample reaction
            threshold = np.random.random() * total_xs
            current = 0.
            for reaction in p.cell.mat.rxn_probs[p.group]:
                reaction_xs = p.cell.mat.rxn_probs[p.group][reaction]
                current += reaction_xs
                if threshold < current:
                    break

            if reaction == 'scatter':
                # Determine outgoing group
                threshold = np.random.random() * reaction_xs
                current = 0.
                for g_out, val in p.cell.mat.xs[p.group]['scatter'].items():
                    current += val
                    if threshold < current:
                        p.group = g_out
                        break

                # Scatter istotropically
                p.dof = p.sample_isotropic()

            elif reaction == 'capture':
                # Score estimators
                capture_score = reaction_xs / total_xs

                for est in p.cell.estimators:
                    estimators[est].score(p.group, capture_score)

                # particle was captured, kill the particle
                p.alive = False

            else:
                raise RuntimeError

        elif min_event == 'surface_crossing':
            # Score estimators
            for est in closest_surface.estimators:
                # score the partial current in +x direction
                if p.dof[0] > 0:
                    estimators[est].score(p.group, 1)

            # Update particle cell
            if (closest_surface.bc == 'vacuum'):
                p.alive = False
            else:
                p.cell = geometry.find_cell(p.pos)
                #print(p.cell.name)

        else:
            raise RuntimeError

        # Tally all of the higher-order statistical moments
    for estimator in estimators.values():
        estimator.end_history()

    return estimators

def init_sim(grps, geom):
    global groups, geometry
    groups = grps
    geometry = geom


def run_histories(runs, procs, tallies=None):
    """Runs runs runs on procs processes."""
    tick = time.perf_counter()

    groups = range(1,2)

    # build the cross section data and material
    xs = {'box' : { 1: { 'scatter': {1: 0.0}, 'capture': 1.0 } } }
    mat = Material(xs['box'])

    # build the two adjacent boxes
    x_minus = Plane('x_minus',1.0,0.0,0.0,-1.0, bc='vacuum')
    x_plus = Plane('x_plus',1.0,0.0,0.0,1.0, bc='vacuum')
    x_zero = Plane('x_zero',1.0, 0.0, 0.0, 0.0)
    y_minus = Plane('y_minus',0.0,1.0,0.0,-1.0, bc='vacuum')
    y_plus = Plane('y_plus',0.0,1.0,0.0,1.0, bc='vacuum')
    z_minus = Plane('z_minus',0.0,0.0,1.0,-1.0, bc='vacuum')
    z_plus = Plane('z_plus',0.0,0.0,1.0,1.0, bc='vacuum')

    box1 = Cell('box1', mat)
    box1.add_surfaces([-x_zero, +x_minus, -y_plus, +y_minus, -z_plus, +z_minus])
    box2 = Cell('box2', mat)
    box2.add_surfaces([-x_plus, +x_zero, -y_plus, +y_minus, -z_plus, +z_minus])

    # build the problem cells, surfaces, and geometry
    cells = {'box1': box1, 'box2': box2}
    surfaces = {'x_minus': x_minus, 'x_plus': x_plus, 'x_zero': x_zero,
                'y_minus': y_minus, 'y_plus': y_plus,
                'z_minus': z_minus, 'z_plus': z_plus}

    geometry = Geometry(cells, surfaces)

    cell = geometry.find_cell([0.5, 0., 0.])
    print(cell.name)

    # Set up the estimators
    x_zero.add_estimator('x_zero_current')
    box1.add_estimator('box1_capture')

    print(f"Running {runs} histories on {procs} processes...")
    sums = { group: {'capture': 0, 'capture sqr': 0,
                     'current': 0, 'current sqr': 0,
                     'fg_leak': 0} for group in groups}

    output = {}

    pool = mp.Pool(processes=procs,initializer=init_sim,initargs=(groups,geometry,))
    results = pool.imap(simulate_particle, range(runs), chunksize=100)

    for result in results:
        for group in groups:
            sums[group]['capture'] += result['box1_capture'].history_sum[group]
            sums[group]['capture sqr'] += result['box1_capture'].history_sum_sq[group]
            sums[group]['current'] += result['x_zero_current'].history_sum[group]
            sums[group]['current sqr'] += result['x_zero_current'].history_sum_sq[group]
            sums[group]['fg_leak'] += result['x_zero_current'].covariance_moment['box1_capture'][group]

    for group in sums:
        capture_mean = sums[group]['capture'] / runs
        capture_mean_square = sums[group]['capture sqr'] / runs
        current_mean = sums[group]['current'] / runs
        current_mean_square = sums[group]['current sqr'] / runs
        fg_leak_mean = sums[group]['fg_leak'] / runs

        if tallies != None:
            rho = tallies[group]['correlation']
            g_mean = tallies[group]['capture']
            f_stdev = tallies[group]['current stdev']
            g_stdev = tallies[group]['capture stdev']
            alpha = -rho * f_stdev / g_stdev
            cv_leak_mean = current_mean + alpha * (capture_mean - g_mean)
            cv_stdev = np.sqrt((current_mean_square + 2 * alpha * (fg_leak_mean - current_mean * g_mean) + alpha**2 * (capture_mean_square - capture_mean**2) - 2 * cv_leak_mean * current_mean + cv_leak_mean ** 2 - 2 * alpha * cv_leak_mean * (capture_mean - g_mean) ) / runs)
        else:
            g_mean = 0
            alpha = 0
            cv_leak_mean = 0
            cv_stdev = 0

        output[group] = {
            'capture': capture_mean,
            'capture stdev': np.sqrt((capture_mean_square - capture_mean ** 2) / runs),
            'current': current_mean,
            'current stdev': np.sqrt((current_mean_square - current_mean ** 2) / runs),
            'correlation': (fg_leak_mean - current_mean * capture_mean) / (np.sqrt(current_mean_square - current_mean ** 2) * np.sqrt(capture_mean_square - capture_mean ** 2)),
            'cv_leak': cv_leak_mean,
            'cv_leak stdev': cv_stdev
        }
    tock = time.perf_counter()
    print(f"time elapsed: {tock-tick:0.4f}")
    return output


if __name__ == '__main__':
    # Run RUNS
    tallies = run_histories(int(1e5), 4)
    print(tallies)
    print(run_histories(int(1e4), 4, tallies))
