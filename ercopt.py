import warnings

from joblib import Parallel, delayed
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.base import clone
from skopt import Optimizer as skopt_Optimizer
from skopt.acquisition import _gaussian_acquisition, gaussian_acquisition_1D
from skopt.utils import is_listlike, is_2Dlistlike

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
    
    def _tell(self, x, y, fit=True):
        """Perform the actual work of incorporating one or more new points.
        See `tell()` for the full description.

        This method exists to give access to the internals of adding points
        by side stepping all input validation and transformation."""

        if "ps" in self.acq_func:
            if is_2Dlistlike(x):
                self.Xi.extend(x)
                self.yi.extend(y)
                self._n_initial_points -= len(y)
            elif is_listlike(x):
                self.Xi.append(x)
                self.yi.append(y)
                self._n_initial_points -= 1
        # if y isn't a scalar it means we have been handed a batch of points
        elif is_listlike(y) and is_2Dlistlike(x):
            self.Xi.extend(x)
            self.yi.extend(y)
            self._n_initial_points -= len(y)
        elif is_listlike(x):
            self.Xi.append(x)
            self.yi.append(y)
            self._n_initial_points -= 1
        else:
            raise ValueError("Type of arguments `x` (%s) and `y` (%s) "
                             "not compatible." % (type(x), type(y)))

        # optimizer learned something new - discard cache
        self.cache_ = {}

        # after being "told" n_initial_points we switch from sampling
        # random points to using a surrogate model
        if (fit and self._n_initial_points <= 0 and
                self.base_estimator_ is not None):
            transformed_bounds = np.array(self.space.transformed_bounds)
            est = clone(self.base_estimator_)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(self.space.transform(self.Xi), self.yi)

            if hasattr(self, "next_xs_") and self.acq_func == "gp_hedge":
                self.gains_ -= est.predict(np.vstack(self.next_xs_))

            if self.max_model_queue_size is None:
                self.models.append(est)
            elif len(self.models) < self.max_model_queue_size:
                self.models.append(est)
            else:
                # Maximum list size obtained, remove oldest model.
                self.models.pop(0)
                self.models.append(est)

            # even with BFGS as optimizer we want to sample a large number
            # of points and then pick the best ones as starting points
            X = self.space.transform(self.space.rvs(
                n_samples=self.n_points, random_state=self.rng))

            self.next_xs_ = []
            for cand_acq_func in self.cand_acq_funcs_:
                values = _gaussian_acquisition(
                    X=X, model=est, y_opt=np.min(self.yi),
                    acq_func=cand_acq_func,
                    acq_func_kwargs=self.acq_func_kwargs)
                # Find the minimum of the acquisition function by randomly
                # sampling points from the space
                if self.acq_optimizer == "sampling":
                    next_x = X[np.argmin(values)]

                # Use BFGS to find the mimimum of the acquisition function, the
                # minimization starts from `n_restarts_optimizer` different
                # points and the best minimum is used
                elif self.acq_optimizer == "lbfgs":
                    x0 = X[np.argsort(values)[:self.n_restarts_optimizer]]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        results = Parallel(n_jobs=self.n_jobs)(
                            delayed(fmin_l_bfgs_b)(
                                gaussian_acquisition_1D, x,
                                args=(est, np.min(self.yi), cand_acq_func,
                                      self.acq_func_kwargs),
                                bounds=self.space.transformed_bounds,
                                approx_grad=False,
                                maxiter=20)
                            for x in x0)

                    cand_xs = np.array([r[0] for r in results])
                    cand_acqs = np.array([r[1] for r in results])
                    next_x = cand_xs[np.argmin(cand_acqs)]

                # lbfgs should handle this but just in case there are
                # precision errors.
                if not self.space.is_categorical:
                    next_x = np.clip(
                        next_x, transformed_bounds[:, 0],
                        transformed_bounds[:, 1])
                self.next_xs_.append(next_x)

            if self.acq_func == "gp_hedge":
                logits = np.array(self.gains_)
                logits -= np.max(logits)
                exp_logits = np.exp(self.eta * logits)
                probs = exp_logits / np.sum(exp_logits)
                next_x = self.next_xs_[np.argmax(self.rng.multinomial(1,
                                                                      probs))]
            else:
                next_x = self.next_xs_[0]

            # note the need for [0] at the end
            self._next_x = self.space.inverse_transform(
                next_x.reshape((1, -1)))[0]

        # Pack and return results
        return self.get_result()