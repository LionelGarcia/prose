import numpy as np
import pymc3 as pm
import theano
from theano import tensor as tt
import exoplanet as xo
from prose.modelling import utils
from functools import reduce


class Model:
    def __init__(self, *args, x=None, name=None, **kwargs):
        self.x = x
        self.default_type="time"
        if isinstance(x, np.ndarray):
            self.x = self.x.astype("float")
        self.name = name

        self.__dict__.update(kwargs)
    
    def is_included(self, i):
        if self.lc is None:
            return True
        elif self.lc == i:
            return True
        elif i in self.lc:
            return True
        else:
            return None


class ConstantModel(Model):
    def __init__(self, x=None,  name="constant", m=0., lc=None):
        super().__init__(x=x,  name=name, m=m, lc=lc)

    def pymc3_model(self):
        x = tt.as_tensor_variable(self.x)
        m = utils.convert_to_pymc3("{}_m".format(self.name), self.m)
        return pm.Deterministic(self.name, tt.ones_like(x) * m)


class LinearModel(Model):
    def __init__(self, x=None, a=1., b=1., name="linear", lc=None):
        super().__init__(x=x, a=a, b=b, name=name, lc=lc)

    def pymc3_model(self):
        x = tt.as_tensor_variable(self.x)
        a = utils.convert_to_pymc3("{}_a".format(self.name), self.a)
        b = utils.convert_to_pymc3("{}_b".format(self.name), self.b)
        return pm.Deterministic(self.name, tt.ones_like(x) * a + self.x * b)


class PolynomialModel(Model):
    def __init__(self, x=None, coeffs=None, name="polynomial", lc=None):
        super().__init__(x=x, coeffs=coeffs, name=name, lc=lc)

    def pymc3_model(self):
        x = tt.as_tensor_variable(self.x)
        coeffs = [utils.convert_to_pymc3("{}_a{}".format(self.name, str(i)), coeff) for i, coeff in
                  enumerate(self.coeffs)]
        s = [tt.pow(x, i) * coeff for i, coeff in enumerate(coeffs)]
        return pm.Deterministic(self.name, tt.sum(tt.stack(s), axis=0))


class Planet(Model):
    """
    Planet model holding its orbital parameters
    
    Parameters
    ----------
    name : [type]
        [description]
    t0 : [type], optional
        [description], by default None
    period : [type], optional
        [description], by default None
    r : [type], optional
        [description], by default None
    b : [type], optional
        [description], by default None
    incl : [type], optional
        [description], by default None
    ecc : [type], optional
        [description], by default None
    omega : [type], optional
        [description], by default None
    a : [type], optional
        [description], by default None
    r_unit : str, optional
        [description], by default "earth"
    """
    def __init__(self, name, t0=0., period=None, r=None, b=None, incl=90.,
                 ecc=None, omega=None, a=None, r_unit="earth", incl_unit="degrees"):
        """
        Planet model holding its orbital parameters
        
        Parameters
        ----------
        name : [type]
            [description]
        t0 : [type], optional
            [description], by default None
        period : [type], optional
            [description], by default None
        r : [type], optional
            [description], by default None
        b : [type], optional
            [description], by default None
        incl : [type], optional
            [description], by default None
        ecc : [type], optional
            [description], by default None
        omega : [type], optional
            [description], by default None
        a : [type], optional
            [description], by default None
        r_unit : str, optional
            [description], by default "earth"
        """
        # TODO:
        # - conversion for incl

        super().__init__(name=name, t0=t0, period=period, r=r, b=b, incl=incl, ecc=ecc,
                         omega=omega, a=a, r_unit=r_unit, incl_unit=incl_unit)

        if self.r_unit == 'earth':
            self.r = utils.convert_distribution(self.r, utils.earth2sun_radius)

        if self.incl_unit == 'degrees':
            self.incl = utils.convert_distribution(self.incl, utils.degrees2radians)

    def __getitem__(self, key):
        return dict(t0=self.t0, period=self.period, r=self.r, b=self.b, incl=self.incl, ecc=self.ecc,
                    omega=self.omega, a=self.a)[key]


class Star(Model):
    def __init__(self, r=None, m=None, u=None):
        """
        Stellar model holding its orbital parameters

        
        Parameters
        ----------
        Model : [type]
            [description]
        r : [type], optional
            [description], by default None
        m : [type], optional
            [description], by default None
        u : [type], optional
            [description], by default None
        """
        super().__init__(name="star", r=r, m=m, u=u)

        # TODO:
        # - conversion for r_star_unit
        # - conversion for m_star_unit

    def __getitem__(self, key):
        return dict(r=self.r, m=self.m, u=self.u)[key]


class OrbitModel(Model):
    def __init__(self, star, planets):
        """
        Orbital model
        Partly from this package and from exoplanet pakage, only specific combinations of input parameters can be used:

        For the star:
            - Two of ``m_star``, ``r_star``, and ``rho_star`` can be defined.

        For the ``Planet``object:
            - An initial distribution of radius ``r`` must be given
            - Either ``period`` or ``a`` must be given. If both are given then neither ``m_star`` or ``rho_star`` can be defined.
            - Either ``incl`` or ``b`` can be given.
            - If a value is given for ``ecc`` then ``omega`` must also be given.
            - Either t0 (reference transit) or t_periastron must be given, not both. *in dev* If ``t0`` is not set, it will be be given a uniform prior in time

        
        Parameters
        ----------
        Model : [type]
            [description]
        star : [type]
            [description]
        """

        super(OrbitModel, self).__init__(name="transit", star=star, planets=planets, x_type="time")

    @property
    def planets_dict(self):
        if len(self.planets) > 1:
            return {planet.name: planet for planet in self.planets}
        else:
            return {"b": self.planets[0]}

    def planets_pymc3_params(self, which, replace_none=None):
        planets_dict = self.planets_dict
        if len(planets_dict) > 1:
            distributions = []
            for planet, params in self.planets_dict.items():
                if replace_none is not None and params is None:
                    distribution = utils.convert_to_pymc3(replace_none)
                else:
                    distribution = utils.convert_to_pymc3("{}_{}".format(which, planet), params[which])
                distributions.append(distribution)
            none_distributions = [d is None for d in distributions]
            if np.any(none_distributions):
                if np.all(none_distributions):
                    return None
                else:
                    raise ValueError("Parameter '{0}', should be set for all planets".format(which))
            else:
                return tt.stack(distributions)
        else:
            return utils.convert_to_pymc3("{}_{}".format(which, "b"), planets_dict["b"][which])

    def pymc3_model(self):

        t0 = self.planets_pymc3_params("t0", replace_none=["u", np.min(self.x), np.max(self.x)])
        period = self.planets_pymc3_params("period")
        r = self.planets_pymc3_params("r")
        b = self.planets_pymc3_params("b")
        incl = self.planets_pymc3_params("incl")
        ecc = self.planets_pymc3_params("ecc")
        omega = self.planets_pymc3_params("omega")
        a = self.planets_pymc3_params("a")

        r_star = utils.convert_to_pymc3("r_star", self.star.r)
        m_star = utils.convert_to_pymc3("m_star", self.star.m)

        # To keep track of the depth
        pm.Deterministic("depth", (r / r_star) ** 2)

        # Orbital model
        orbit = xo.orbits.KeplerianOrbit(
            period=period,
            t0=t0,
            b=b,
            r_star=r_star,
            m_star=m_star,
            omega=omega,
            ecc=ecc,
            incl=incl,
            a=a
        )

        return orbit, r

        # # Compute the model light curve using starry
        # light_curves = xo.LimbDarkLightCurve(u).get_light_curve(
        #     orbit=orbit, r=r, t=self.x.astype("float")
        # )

        # # Here we track the value of individual transit light curves for plotting purposes
        # _ = pm.Deterministic("light_curves", light_curves)

        # return pm.Deterministic(self.name, pm.math.sum(light_curves, axis=-1))


class SimpleFlareModel(Model):
    def __init__(self, x=None, t0=0., tau=1., amp=1., name="flare", lc=None):
        super().__init__(x=x, t0=t0, tau=tau, amp=amp, name=name, lc=lc)

    def pymc3_model(self):
        t0 = utils.convert_to_pymc3("t0_flare", self.t0)
        tau = utils.convert_to_pymc3("tau_flare", self.tau)
        amp = utils.convert_to_pymc3("amp_flare", self.amp)
        time = tt.as_tensor_variable(self.x)

        before = tt.zeros_like(time)
        after = amp * tt.exp(-(time - t0)/tau)

        return pm.Deterministic(self.name, tt.switch(tt.lt(time, t0), before, after))

    def set_parameter(self, t0=None, tau=None, amp=None):
        if t0 is not None:
            self.t0 = t0
        if tau is not None:
            self.tau = tau
        if amp is not None:
            self.amp = amp

    def model(self):
        time_before = self.x[self.x < self.t0]
        time_after = self.x[self.x >= self.t0]

        before = np.zeros_like(time_before)
        after = self.amp * np.exp(-(time_after - self.t0) / self.tau)

        return (self.name, np.hstack([before, after]))
    
    
class PolynomialSystematics(Model):
    def __init__(self, x=None, name="polynomial_systematics", orders={}, lc=None):
        super().__init__(x=x, name=name, orders=orders, lc=lc)
        self.default_type="systematics"
    
    def build_X(self):
        n = len(list(self.x.values())[0])
        
        def rescale(x):
            return (x - np.mean(x)) / np.std(x)
        
        X = [np.ones(n)]
        
        for field, order in self.orders.items():
            for o in range(1, order + 1):
                X.append(np.power(rescale(self.x[field]).astype("float"), o))
                
        self.X = np.array(X).transpose()
        
    def pymc3_model(self, y):
        self.build_X()
        tX = tt.as_tensor_variable(self.X)
        ty = y
        
        svd = tt.nlinalg.SVD(full_matrices=False)

        U, S, V = svd(tX)
        S = tt.diag(S)
        S_ = tt.set_subtensor(S[tt.eq(S, 0.)], 1e10)

        coeffs = pm.Deterministic("{}_coeffs".format(self.name), reduce(tt.dot, [U.T, ty.T, 1.0 / S_, V]))
        
        return pm.Deterministic(self.name, tt.dot(coeffs, tX.T))