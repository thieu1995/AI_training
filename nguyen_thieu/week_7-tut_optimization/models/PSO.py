import numpy as np
import numpy.random as np_rand

from time import sleep
import matplotlib.pyplot as plt

# try:
#     from IPython.parallel.client.view import LoadBalancedView
#     from multiprocessing.pool import Pool
# except:
#     pass


import collections

# Allow instance methods to be used as objective functions
# from . import pickle_method

import logging

# logger.basicConfig(format='%(levelname)s: %(message)s', level=logger.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
_handler.setFormatter(_formatter)
logger.addHandler(_handler)


def set_log_level(level):
    """ Allow quick setting of module-level logging. """
    logger.setLevel(level)
    logger.handlers[0].setLevel(level)


class PartFuncObj(object):
    def __init__(self, f, *args):
        """Initialise a partial function object using a function and a set of
        invariant arguments"""
        self.f = f
        self.fixed_args = list(args)

    def __call__(self, vari_args_iterable):
        """When instance called as function run the enclosed function with
        static and per-call args"""
        return self.f(*list(vari_args_iterable) + self.fixed_args)


class PSO(object):
    """
    Particle Swarm Optimisation implementation

    References:
    Eberhart, R.C., Shi, Y., 2001. Particle swarm optimization: developments,
    applications and resources, in: Proceedings of the 2001 Congress on
    Evolutionary Computation. Presented at the Congress on Evolutionary
    Computation, pp. 81-86.

    Kennedy, J., Mendes, R., 2002. Population structure and particle swarm
    performance. IEEE, pp. 1671-1676.

    Xu, S., Rahmat-Samii, Y., 2007. Boundary conditions in particle swarm
    optimization revisited. IEEE Transactions on Antennas and Propagation,
    55, 760-765.

    """

    def __init__(self, obj_func, box_bounds, n_parts=5, topo="gbest",
                 weights=[0.9, 0.4, 2.1, 2.1], extra_args=None,
                 minimise=True, parallel_arch=None):
        """Initialise the positions and velocities of the particles, the
        particle memories and the swarm-level params.

        Keyword args:
        obj_func -- Objective function (ref stored in the attribute
                    'orig_obj_func')
        box_bounds -- tuple of (lower_bound, upper_bound) tuples (or
                  equivalent ndarray), the length of the former being the
                  number of dimensions e.g. ((0,10),(0,30)) for 2 dims.
                  Restricted damping is used (Xu and Rahmat-Samii, 2007) when
                  particles go out of bounds.
        n_parts -- number of particles
        topo -- 'gbest', 'ring' or 'von_neumann'
        weights -- 4-vector of weights for inertial (initial and final),
                   nostalgic and societal velocity components
        extra_args -- dictionary of keyword arguments to be passed to the
                    objective function; these arguments do not correspond to
                    variables that are to be optimised
        minimise -- whether to find global minima or maxima for the objective
                    function
        parallel_arch -- fitness function evaluation during swarm instantiation
                         can optionally be done using master/slave
                         parallelisation if parallel_arch is a
                         IPython.parallel.client.view.LoadBalancedView or
                         multiprocessing.pool.Pool instance.

        Notes on swarm and neighbourhood sizes (Eberhart and Shi, 2001)
        Swarm size of 20-50 most common.
        Neighbourhood size of ~15% swarm size used in many applications.

        Notes on topologies:
        Two neighbourhood topologies have been implemented (see Kennedy and
        Mendes, 2002).  These are both social rather than geographic
        topologies and differ in terms of degree of connectivity (number of
        neighbours per particle) but not in terms of clustering (number of
        common neighbours of two particles).  The supported topologies are:
        ring -- exactly two neighbours / particle
        von_neumann -- exactly four neighbours / particle (swarm size must
                       be square number)
        gbest -- the neighbourhood of each particle is the entire swarm

        Notes on weights (Eberhart and Shi, 2001, Xu and Rahmat-Samii, 2007):
        inertia weight   -- decrese linearly from 0.9 to 0.4 over a run
        nostalgia weight -- 2.05
        societal weight -- 2.05
        NB sum of nostalgia and societal weights should be >4 if using
        Clerc's constriction factor.

        Notes on parallel objective function evaluation:
        This may not be efficient if a single call to the objective function
        takes very little time to execute.
        """

        # Whether to minimise or maximise the objective function
        self.minimise = minimise

        # Store refs to the original objective function and extra_args
        self.orig_obj_func = obj_func
        self.extra_args = tuple(extra_args) if extra_args else None

        # If the objective function doesn't requires additional arguments that
        # don't correspond to the dimensions of the problem space:
        if not extra_args or len(extra_args) < 1:
            # Create a vectorized version of that function
            # for fast single-threaded execution
            self.obj_func_vectorized = np.vectorize(self.orig_obj_func)
            # Create new reference for multi-threaded evaluation
            # using a multiprocessing.pool.Pool workpool
            self.obj_func = self.orig_obj_func
        else:
            # Create a vectorized, partial form of that function
            # for fast single-threaded evaluation
            self.obj_func_vectorized = np.vectorize(lambda args:
                                                    self.orig_obj_func(*(args + self.extra_args)))
            # Also create a partial form of that function using a function
            # object for multi-threaded evaluation using a
            # multiprocessing.pool.Pool workpool
            # self.obj_func = PartFuncObj(self.orig_obj_func, *self.extra_args)
            self.obj_func_vectorized = np.vectorize(
                lambda *args: self.orig_obj_func(*args + tuple(extra_args)))

            # TESTING
            self.obj_func_test = lambda args: self.orig_obj_func(*(args +
                                                                   self.extra_args))

        # Initialise velocity weights
        self.set_weight(weights)

        # Determine the problem space boundaries
        # NB: should really parse box_bounds to ensure that it is valid
        self.lower_bounds, self.upper_bounds = np.asfarray(box_bounds).T
        if np.any(self.lower_bounds >= self.upper_bounds):
            raise ValueError("All lower bounds must be < upper bounds")

        # Set number of particles
        self._n_parts = n_parts

        # Set number of dimensions
        self._n_dims = len(box_bounds)

        # Initialise particle positions
        # Each row is the position vector of a particular particle
        self.pos = np_rand.uniform(self.lower_bounds,
                                   self.upper_bounds,
                                   (self._n_parts, self._n_dims))

        # Previous best position for all particles is starting position
        self.pbest = self.pos.copy()

        # Initialise velocity matrix
        # Each row is the velocity vector of a particular particle
        self.vel = np_rand.uniform(- (self.upper_bounds - self.lower_bounds),
                                   self.upper_bounds - self.lower_bounds,
                                   (self._n_parts, self._n_dims))

        # Check parallel_arch is supported and return type
        self.parallel_arch_type = _get_parallel_arch_type(parallel_arch)

        # Find the performance per particle
        # (updates self.perf, a matrix of self._n_parts rows and
        # self._n_dims cols)
        self._eval_perf(parallel_arch)

        # The personal best position per particle is initially the starting
        # position
        self.pbest_perf = self.perf.copy()

        # Initialise swarm best position (array of length self._n_dims)
        # and the swarm best performance  (scalar)
        if self.minimise:
            self.swarm_best = self.pos[self.perf.argmin()]
            self.swarm_best_perf = self.perf.min()
        else:
            self.swarm_best = self.pos[self.perf.argmax()]
            self.swarm_best_perf = self.perf.max()

        if logger.level < logging.INFO:
            logger.debug("INIT POS:  %s", self.pos)
            logger.debug("INIT PERF: %s", self.perf)
            logger.debug("INIT VEL:  %s", self.vel)

        # Determine particle neighbours if a swarm topology has been chosen
        self._cache_neighbourhoods(topo)

    def set_weight(self, w_inertia_start=None, w_inertia_end=None,
                   w_nostalgia=None, w_societal=None):
        """
        Set velocity component weights

        Keyword arguments:
        w_inertia_start -- particle inertia weight at first iteration
                           (recommended value: 0.9)
        w_inertia_end -- particle inertia weight at maximum possible iteration
                         (recommended value: 0.4)
        w_nostalgia -- weight for velocity component in direction of particle
                       historical best performance (recommended value: 2.1)
        w_societal -- weight for velocity component in direction of current
                      best performing particle in neighbourhood (recommended
                      value: 2.1)

        NB can set all four weights simultanously by passing the method an
        iterable object of length 4 e.g.:

        weights = [0.9, 0.4, 2.1, 2.1]
        my_pyshoal.set_weight(weights)
        """
        if isinstance(w_inertia_start, collections.Iterable) and \
                len(w_inertia_start) == 4 and None not in w_inertia_start:
            self.w_inertia_start, self.w_inertia_end, \
            self.w_nostalgia, self.w_societal = w_inertia_start
        else:
            if w_inertia_start is not None:
                self.w_inertia_start = w_inertia_start
            if w_inertia_end is not None:
                self.w_inertia_end = w_inertia_end
            if w_nostalgia is not None:
                self.w_nostalgia = w_nostalgia
            if w_societal is not None:
                self.w_societal = w_societal

    def _eval_perf(self, parallel_arch=None):
        if parallel_arch is None:
            # self.perf = self.obj_func_vectorized(*self.pos.T)
            self.perf = self.orig_obj_func(self.pos.T)

        elif self.parallel_arch_type == 'multiprocessing.pool':
            if self._n_dims > 1:
                self.perf = np.array(parallel_arch.map(self.obj_func,
                                                       self.pos.T))
            else:
                # Need to package self.pos here as one-element iterable?
                self.perf = np.array(parallel_arch.map(self.obj_func, self.pos))
        # TESTING START
        elif self.parallel_arch_type == 'IPython.parallel':
            # self.perf = parallel_arch.apply_sync(self.obj_func,
            #                                     *self.pos.T)
            # parallel_arch.scatter('pos', self.pos)
            # parallel_arch.push({'obj_func': self.obj_func})
            self.perf = np.array(parallel_arch.map(self.obj_func, *self.pos.T))
            # TESTING END
        else:
            raise Exception("Invalid parallel architecture")

    def _cache_neighbourhoods(self, topo):
        """Determines the indices of the neighbours per particle and stores them
        in the neighbourhoods attribute.

        Currently only the following topologies are supported (see Kennedy and
        Mendes, 2002):

        'gbest' - Global best (no local particle neighbourhoods)
        'von_neumann' -- Von Neumann lattice (each particle communicates with
                         four social neighbours)
        'ring' -- Ring topology (each particle communicates with two social
                  neighbours)

        """
        self.topo = topo
        if self.topo == "gbest":
            return
        if self.topo not in ("von_neumann", "ring"):
            raise Exception("Topology '{}' not recognised/supported".format(
                self.topo))

        n = self._n_parts

        if self.topo == "von_neumann":
            # Check that p_idx is square
            n_sqrt = int(np.sqrt(n))
            if not n_sqrt ** 2 == n:
                raise Exception("Number of particles needs to be perfect " +
                                "square if using Von Neumann neighbourhood " +
                                "topologies.")
            self.neighbourhoods = np.zeros((n, 5), dtype=int)
            for p in range(n):
                self.neighbourhoods[p] = [
                    p,  # particle
                    (p - n_sqrt) % n,  # p'cle above
                    (p + n_sqrt) % n,  # p'cle below
                    ((p // n_sqrt) * n_sqrt) + ((p + 1) % n_sqrt),  # p'cle to r
                    ((p // n_sqrt) * n_sqrt) + ((p - 1) % n_sqrt)  # p'cle to l
                ]
        elif self.topo == "ring":
            self.neighbourhoods = np.zeros((n, 3), dtype=int)
            for p in range(n):
                self.neighbourhoods[p] = [
                    (p - 1) % n,  # particle to left
                    p,  # particle itself
                    (p + 1) % n  # particle to right
                ]

    def _velocity_updates(self, itr, max_itr, use_constr_factor=False):
        """Update particle velocities.

        Keyword arguments:
        itr     -- current timestep
        max_itr -- maximum number of timesteps
        use_constr_factor -- whether Clerc's constriction factor should be
                             applied

        New velocities determined using
         - the supplied velocity factor weights
         - random variables to ensure the process is not deterministic
         - the current velocity per particle
         - the best performing position in each particle's personal history
         - the best performing current position in each particle's neighbourhood

        Max velocities clamped to length of problem space boundaries and
        Clerc's constriction factor applied as per Eberhart and Shi (2001)

        Spits out the following if logging level is INFO
         - best neighbours per particle
         - velocity components per particle
         - velocities per particle

        NB should only be called from the VCDM class's _tstep method.
        """
        w_inertia = self.w_inertia_start + \
                    (self.w_inertia_end - self.w_inertia_start) * \
                    (itr / float(max_itr))

        inertia_vel_comp = w_inertia * self.vel
        nostalgia_vel_comp = self.w_nostalgia * np_rand.rand() * \
                             (self.pbest - self.pos)
        societal_vel_comp = self.w_societal * np_rand.rand() * \
                            (self.best_neigh - self.pos)

        self.vel = inertia_vel_comp + nostalgia_vel_comp + societal_vel_comp

        # Constriction factor
        if use_constr_factor:
            phi = self.w_nostalgia + self.w_societal
            if not phi > 4:
                raise Exception("Cannot apply constriction factor as sum of " +
                                "societal and nostalgic weights <= 4")
            K = 2. / np.abs(2. - phi - np.sqrt((phi ** 2) - (4. * phi)))
            self.vel *= K

        # Velocity clamping
        self.vel.clip(self.lower_bounds - self.upper_bounds,
                      self.upper_bounds - self.lower_bounds,
                      out=self.vel)

        if logger.level < logging.INFO:
            logger.debug("BEST NEIGHBOURS:   {}".format(self.best_neigh))
            logger.debug("INERTIA V COMP:    {}".format(inertia_vel_comp))
            logger.debug("NOSTALGIA V COMP:  {}".format(nostalgia_vel_comp))
            logger.debug("SOCIETAL V COMP:   {}".format(societal_vel_comp))
            logger.debug("VELOCITIES:        {}".format(self.vel))

    def _box_bounds_checking(self):
        """Apply restrictive damping if position updates have caused particles
        to leave problem space boundaries.

        Restrictive damping explained in:
        Xu, S., Rahmat-Samii, Y. (2007). Boundary conditions in particle swarm
        optimization revisited. IEEE Transactions on Antennas and Propagation,
        vol 55, pp 760-765.

        """
        too_low = self.pos < self.lower_bounds
        too_high = self.pos > self.upper_bounds

        # Ensure all particles within box bounds of problem space
        self.pos.clip(self.lower_bounds, self.upper_bounds, out=self.pos)

        old_vel = self.vel.copy()
        self.vel[too_low | too_high] *= \
            -1. * np_rand.random((self.vel[too_low | too_high]).shape)

        if np.array_equal(old_vel, self.vel) and np.any(too_low | too_high):
            raise Exception("Velocity updates in boundary checking code " +
                            "not worked")

        # What if reflected particles now comes out the other side of the
        # problem space? TODO
        too_low = self.pos < self.lower_bounds
        too_high = self.pos > self.upper_bounds
        if np.any(too_low) or np.any(too_high):
            raise Exception("Need multiple-pass bounds checking")

    def _tstep(self, itr, max_itr, parallel_arch=None):
        """Optimisation timestep function

        Keyword arguments:
        itr -- current timestep
        max_itr -- maximum number of timesteps
        parallel_arch -- IPython.parallel.client.view.LoadBalancedView
                         or multiprocessing.Pool instance for parallel objective
                         function evaluation.
        """
        if self.topo in ("von_neumann", "ring"):
            # For each particle, find the 'relative' index of the best
            # performing neighbour e.g. 0, 1 or 2 for particles in ring
            # topologies
            if self.minimise:
                best_neigh_idx = self.neighbourhoods[np.arange(self._n_parts),
                                                     self.perf[self.neighbourhoods].argmin(axis=1)]
            else:
                best_neigh_idx = self.neighbourhoods[np.arange(self._n_parts),
                                                     self.perf[self.neighbourhoods].argmax(axis=1)]
            # then generate a vector of the _positions_ of the best performing
            # particles in each neighbourhood (the length of this vector will be
            # equal to the number of particles)
            self.best_neigh = self.pos[best_neigh_idx]

        else:
            # self.topo == "gbest". For the 'global best' approach the best
            # performing neighbour is the best performing particle in the entire
            # swarm
            self.best_neigh = self.swarm_best

        # Update the velocity and position of each particle
        self._velocity_updates(itr, max_itr)
        self.pos += self.vel

        # Check that all particles within problem boundaries;
        # if not move to edges of prob space and
        # flip signs of and dampen velocity components that took particles
        # over boundaries
        self._box_bounds_checking()

        # Cache the current performance per particle
        self._eval_perf(parallel_arch)

        # Update each particle's personal best position if an improvement has
        # been made this timestep
        if self.minimise:
            improvement_made_idx = self.perf < self.pbest_perf
        else:
            improvement_made_idx = self.perf > self.pbest_perf
        self.pbest[improvement_made_idx] = self.pos[improvement_made_idx]
        self.pbest_perf[improvement_made_idx] = self.perf[improvement_made_idx]

        # Update swarm best position with current best particle position
        if self.minimise:
            self.swarm_best = self.pos[self.perf.argmin()]
            self.swarm_best_perf = self.perf.min()
        else:
            self.swarm_best = self.pos[self.perf.argmax()]
            self.swarm_best_perf = self.perf.max()

        if logger.level < logging.INFO:
            logger.debug("NEW POS:           {}".format(self.pos))
            logger.debug("NEW PERF:          {}".format(self.perf))
            logger.debug("SWARM BEST POS:    {}".format(self.swarm_best))
            logger.debug("SWARM BEST PERF:   {}".format(self.swarm_best_perf))

        return self.swarm_best, self.swarm_best_perf

    def plot_swarm(self, itr, xlims, ylims, contours_delta=None, sleep_dur=0.1,
                   save_plots=False):
        """Plot the swarm in 2D along with performance vs iterations.

        Keyword arguments:
        itr -- current interation (integer)
        xlims -- tuple of min and max x bounds to use for plotting the swarm
        ylims -- tuple of min and max y bounds to use for plotting the swarm
        contours_delta -- delta used for producing contours of objective
                          function. Set to 'None' to disable contour plotting.
        sleep_dur -- time to sleep for between iterations / plots (seconds)
        save_plots -- save plots as a series of png images if True

        """
        if itr == 0:
            # Initialise plots and cache contour data
            if self._n_dims != 2 or len(xlims) != 2 or len(ylims) != 2:
                raise Exception("Cannot plot swarm if the number of " +
                                "dimensions is not 2.")

            self._plt_fig = plt.figure(figsize=(10, 5))
            self._plt_swarm_ax = self._plt_fig.add_subplot(1, 2, 1)
            self._plt_perf_ax = self._plt_fig.add_subplot(1, 2, 2)
            if contours_delta:
                x = np.arange(xlims[0], xlims[1], contours_delta)
                y = np.arange(ylims[0], ylims[1], contours_delta)
                self._contour_X, self._contour_Y = np.meshgrid(x, y)
                self._contour_Z = self.obj_func_vectorized(self._contour_X,
                                                           self._contour_Y)
        else:
            self._plt_swarm_ax.clear()
            self._plt_perf_ax.clear()

        self._plt_swarm_ax.set_xlim(xlims)
        self._plt_swarm_ax.set_ylim(ylims)

        # Plot contours
        if contours_delta:
            cplot = self._plt_swarm_ax.contour(self._contour_X, self._contour_Y,
                                               self._contour_Z)
            self._plt_swarm_ax.clabel(cplot, inline=1, fontsize=10)

        # Plot swarm
        self._plt_swarm_ax.scatter(self.pos.T[0], self.pos.T[1])
        self._plt_perf_ax.plot(self.swarm_best_perf_hist)
        self._plt_perf_ax.set_xlim(xmax=itr)

        # Add title
        title = "Parts: {}   Dims: {}\nTopo: {}   ".format(self._n_parts,
                                                           self._n_dims,
                                                           self.topo)
        self._plt_swarm_ax.set_title(title)
        title = "Best pos: {}\nBest perf: {:5.5f}".format(self.swarm_best,
                                                          self.swarm_best_perf)
        self._plt_perf_ax.set_title(title)
        # plt.subplots_adjust(top=0.80)
        self._plt_swarm_ax.set_xlabel("Var 1")
        self._plt_swarm_ax.set_ylabel("Var 2")
        self._plt_perf_ax.set_xlabel("Iter (current: {})".format(itr))

        plt.draw()
        if save_plots:
            self._plt_fig.savefig("%03d.png" % itr)
        sleep(sleep_dur)

    def opt(self, max_itr=100, tol_thres=None, tol_win=5, parallel_arch=None,
            plot=False, save_plots=False, callback=None):
        """Attempt to find the global optimum objective function.

        Keyword args:
        max_itr -- maximum number of iterations
        tol_thres -- convergence tolerance vector (optional); length must be
                     equal to the number of dimensions.  Can be ndarray, tuple
                     or list.
        tol_win -- number of timesteps for which the swarm best position must be
                   less than convergence tolerances for the funtion to then
                   return a result
        parallel_arch -- fitness function evaluation at each PSO timestep
                         can optionally be expedited using master/slave
                         parallelisation if parallel_arch is either a
                         IPython.parallel.client.view.LoadBalancedView
                         instance or a multiprocessing.Pool instance.

        callback -- callback function that is executed per timestep as
                    'callback(self, itr)'

        Returns:
         -- swarm best position
         -- performance at that position
         -- iterations taken to converge

        """
        if plot:
            plt.ion()

        self.swarm_best_hist = np.zeros((max_itr, self._n_dims))
        self.swarm_best_perf_hist = np.zeros((max_itr,))

        self.tol_thres = np.asfarray(tol_thres)
        # if self.tol_thres.shape != (self._n_dims,):
        #     raise Exception("The length of the tolerance vector must be " +
        #                     "equal to the number of dimensions")

        itr = 0
        while itr < max_itr:
            # Run timestep code
            self.swarm_best_hist[itr], self.swarm_best_perf_hist[itr] = \
                self._tstep(itr, max_itr, parallel_arch)
            if callback is not None:
                callback(self, itr)

            if plot:
                self.plot_swarm(itr, xlims=(-50, 50), ylims=(-50, 50),
                                contours_delta=1., sleep_dur=0.001,
                                save_plots=save_plots)

            # If the convergence tolerance has been reached then return
            # if self.tol_thres is not None and itr > tol_win:
            #     win = self.swarm_best_hist[itr-tol_win+1:itr+1]
            #     if np.all(win.max(axis=0) - win.min(axis=0) < self.tol_thres):
            #         logger.debug("PSO converged in {} iterations.".format(itr))
            #         break
            itr += 1
            # if itr >= max_itr:
            #     logger.info("PSO failed to converge within " +
            #                 "{} iterations.".format(max_itr))

        if plot:
            plt.ioff()
            plt.show()
        return self.swarm_best, self.swarm_best_perf, itr


# END class PSO


def _get_parallel_arch_type(p_arch):
    if p_arch is None:
        return None
    fqcn = p_arch.__module__ + '.' + p_arch.__class__.__name__
    # if fqcn == 'IPython.parallel.client.view.LoadBalancedView':
    if fqcn == 'IPython.parallel.client.view.DirectView':
        parallel_arch_type = 'IPython.parallel'
    elif fqcn == 'multiprocessing.pool.Pool':
        parallel_arch_type = 'multiprocessing.pool'
    else:
        raise Exception("Parallel architecture 'fqcn' not supported")
    return parallel_arch_type


def obj_func_1(x, y, z):
    return 3 * (x - y) ** 4 + 4 * (x - z) ** 2 / (1 + x ** 2 + y ** 2 + z ** 2) + \
           np.cosh(x - 1) + np.tanh(y + z)


def obj_func_quadratic(x):
    return 3 - ((12 + x) ** 2)


def obj_func_rastrigin(a, b):
    return (10 * 2 +
            (a ** 2 - (10 * np.cos(2 * np.pi * a))) +
            (b ** 2 - (10 * np.cos(2 * np.pi * b))))


def obj_func_sinc(a, b):
    return np.sinc(a) + np.sinc(b)


def obj_func_parabola(a, b):
    return a ** 2 + b ** 2


def abc(x):
    output = 0
    for i in range(0, len(x), 2):
        output += x[i] ** 2 + x[i + 1] ** 3
    return output


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print("Need objective function name as argument.")
    #     sys.exit(-1)
    # if sys.argv[1] == 'quadratic':
    #     obj_func = obj_func_quadratic
    # elif sys.argv[1] == 'rastrigin':
    #     obj_func = obj_func_rastrigin
    # elif sys.argv[1] == 'sinc':
    #     obj_func = obj_func_sinc
    # elif sys.argv[1] == 'parabola':
    #     obj_func = obj_func_parabola
    # else:
    #     raise Exception("Objective func {} not supported.".format(sys.argv[1]))
    box_bounds = ()
    tol_thres = ()
    for i in range(50):
        box_bounds += ((-10, 10),)
        tol_thres += ((0.01, 0.01),)
    # print(box_bounds)
    o = PSO(obj_func=abc, box_bounds=box_bounds, n_parts=25,
            topo="von_neumann", weights=[0.9, 0.4, 2.1, 2.1], minimise=True)
    res = o.opt(max_itr=5000, tol_thres=tol_thres, tol_win=5, plot=False,
                save_plots=False)
    logger.info("\nBest position: {}\nBest perf: {}\nNum iter: {}\n".format(*res))