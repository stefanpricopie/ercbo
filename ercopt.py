import warnings

from skopt import Optimizer as skopt_Optimizer

class Optimizer(skopt_Optimizer):
    def __init__(self, erc, strategy, *args, **kwargs):
        # add erc to the list of arguments
        super().__init__(*args, **kwargs)
        self.erc = erc
        self.strategy = strategy
    
    def _ask(self):
        """Suggest next point at which to evaluate the objective.

        Return a random point while not at least `n_initial_points`
        observations have been `tell`ed, after that `base_estimator` is used
        to determine the next point.
        """
        if self._n_initial_points > 0 or self.base_estimator_ is None:
            # _n_initial_points is decremented in `tell`
            # if _n_initial_points > 0, then the initial set of point is not complete
            # if base_estimator_ is None, then the base_estimator has not been fit yet

            if self._initial_samples is not None:
                raise NotImplementedError("Set initial_point_generator to 'random'")

            # initial points are unaffected by the constraint - sample unconstrained
            next_x = self.erc.sample(constrained=False)

            # evaluation will be performed unconstrained at the next time step
            self.erc.t += 1

        else:
            if not self.models:
                raise RuntimeError("Random evaluations exhausted and no "
                                   "model has been fit.")

            next_x = self._next_x
            min_delta_x = min([self.space.distance(next_x, xi)
                               for xi in self.Xi])
            if abs(min_delta_x) <= 1e-8:
                warnings.warn("The objective has been evaluated "
                              "at this point before.")

            # get time step of evaluation
            if self.strategy != 'commitment' and self.erc.active and not self.erc.in_schema(next_x):
                # simulate waiting by deactivating constraint and skipping time steps
                # deactivate constraint
                self.erc.deactivate()

                # evaluation will be performed after epoch has ended (t mod epoch == 1)
                self.erc.t += 1 + self.erc.epoch - self.erc.t % self.erc.epoch
            else:
                # evaluation will be performed unconstrained at the next time step
                self.erc.t += 1

        # return point computed from last call to tell()
        return next_x