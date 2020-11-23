import numpy as np
import pymc3 as pm
import exoplanet as xo
from functools import reduce
import prose.modelling.utils as utils
from theano import tensor as tt
from prose.modelling.models import OrbitModel


class SuperList:
    def __init__(self):
        self.models = []

    def __iadd__(self, other):
        self.models.append(other)
        return self

    def __len__(self):
        return len(self.models)

    def __iter__(self):
        return iter(self.models)

    def __getitem__(self, item):
        return self.models[item]


class HMC:

    def __init__(self, time=None, systematics=None, **kwargs):
        self.pm_model = None
        self.optimized = None
        self.trace = None

        self.models = SuperList()
        self.residuals_model = None
        
        if isinstance(time, np.ndarray):
            self.time = time.astype("float")
    
        self.systematics = systematics

        self.star = SuperList()
        self.bodies = SuperList()

        self.lcs = SuperList()

    @property
    def y_optimized(self):
        if self.optimized is None:
            raise ValueError("Model is not optimized")

        return self.optimized["observation"]

    @property
    def y_sampled(self):
        if self.trace is None:
            raise ValueError("Model is not sampled")
        return np.mean(self.trace["observation"], axis=0)

    def get_lc(self, i=None):
        pass

        # if i is None:
        #     i = range(len(self.lcs))

        # # Be carrefyl that if model is globa
        # lc = np.hstack([flux], )

    def _fill_variable(self, model, lc_i):
        if model.x is None:
            if model.default_type == "time":
                model.x = self.lcs[lc_i][0].astype("float")
            elif model.default_type == "systematics":
                if len(self.lcs[lc_i]) < 4:
                    raise ValueError("light curve {} does not have systematics data".format(lc_i))
                else:
                    model.x = self.lcs[lc_i][3]
            else:
                raise ValueError("'{}' not a correct type".format(model.default_type))
            
        
    def _get_models(self, lc_i):
        models = []
        for _model in self.models:
            if _model.is_included(lc_i):
                self._fill_variable(_model, lc_i)
                _model.name = "{}_{}".format(_model.name, str(lc_i))
                models.append(_model)
        return models

    def build_model(self, optimize=True, verbose=False):

        with pm.Model() as model:
            orbit, r = OrbitModel(self.star, self.bodies).pymc3_model()
            u = self.star.u

            if u is None:
                u = xo.distributions.QuadLimbDark("u", testval=np.array([0.3, 0.2]))
            else:
                u = pm.Deterministic("u", tt.as_tensor_variable(u))

            likelihoods = []

            for i, lc in enumerate(self.lcs):
                if len(lc) == 3:
                    time, flux, error = lc
                    data = None
                elif len(lc) == 4:
                    time, flux, error, data = lc

                # Compute the model light curve using starry
                light_curves = xo.LimbDarkLightCurve(u).get_light_curve(
                    orbit=orbit, r=r, t=time.astype("float"))
                # Here we track the value of individual transit light curves for plotting purposes
                pm.Deterministic("transit_lcs_" + str(i), light_curves)
                pm_transit_model = pm.Deterministic("transit_" + str(i), pm.math.sum(light_curves, axis=-1))

                pm_models = [m.pymc3_model() for m in self._get_models(i)]
                pm_models.append(pm_transit_model)
                pm_lc_model = pm.Deterministic(
                    "lc_model_" + str(i), 
                    reduce(lambda a, b: a + b, pm_models))
                
                if self.residuals_model is not None:
                    if self.residuals_model.is_included(i):
                        if data is None:
                            raise ValueError("light curve {} does not have systematics data".format(i))
                        self._fill_variable(self.residuals_model, i)
                        self.residuals_model.name += "_{}".format(str(i))
                        residuals = tt.as_tensor_variable(flux.astype("float")) - pm_lc_model
                        pm_residual_model = self.residuals_model.pymc3_model(residuals)
                        observation = pm.Deterministic("observation_" + str(i), pm_lc_model+pm_residual_model)
                    else:
                        observation = pm.Deterministic("observation_" + str(i), pm_lc_model)
                else:
                    observation = pm.Deterministic("observation_" + str(i), pm_lc_model)

                # Likelihood assuming known Gaussian uncertainty
                pm.Normal("likelihood_" + str(i), mu=observation, sd=error,  observed=flux)
            
            self.pm_model = model

        if optimize:
            self.optimize(verbose)
    
    @property
    def models_dict(self):
        return {model.name: model for model in self.models}

    def optimize(self, verbose=False):
        with self.pm_model as model:
            self.optimized = xo.optimize(start=model.test_point, verbose=verbose)

    def sample(self, tune=3000, draws=3000, cores=None, target_accept=0.85, chains=2):
        np.random.seed()
        with self.pm_model:
            self.trace = pm.sample(
                tune=tune,
                draws=draws,
                start=self.optimized,
                cores=cores,
                chains=chains,
                step=xo.get_dense_nuts_step(target_accept=target_accept))

    def eval_model(self, name, state):

        if state == "optimized":
            return self.optimized[name]
        elif state == "sampled":
            return utils.mean_trace(self.trace)[name]
        else:
            raise ValueError("state must be 'optimized' or 'sampled', not '{}'".format(state))

    def eval_global_model(self):
        return reduce(lambda a, b: a + b, [self.eval_model(model) for model in self.models_dict.keys()])
