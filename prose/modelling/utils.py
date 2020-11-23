import numpy as np
import pymc3 as pm
from astropy import constants as c
from theano import tensor as tt


def degrees2radians(w):
    return w/180*np.pi

def earth2sun_radius(r):
    return (r * c.R_earth / c.R_sun).value


def mean_trace(traces, method=lambda x: np.mean(x, axis=0)):
    _traces = {}
    for key in traces[0].keys():
        _traces[key] = method([trace[key] for trace in traces])

    return _traces


def convert_distribution(distribution, method=lambda x: x):
    plusminus_chars = ["±", "+-"]
    available_names = ["u", "n", "d"]

    b = None
    test_value = None

    if isinstance(distribution, list):
        if isinstance(distribution[0], str):
            if distribution[0] in available_names:
                name = distribution[0]
                if distribution[0] in ["u", "n"]:
                    assert len(distribution) >= 3, "distribution {} not understood".format(distribution)
                    a = distribution[1]
                    b = distribution[2]
                    if len(distribution) == 4:
                        test_value = distribution[3]
                elif distribution[0] == "d":
                    assert len(distribution) >= 2, "distribution {} not understood".format(distribution)
                    a = distribution[1]
            elif np.any([c in distribution[0] for c in plusminus_chars]):
                name = "n"
                for c in plusminus_chars:
                    if c in distribution[0]:
                        a, b = distribution[0].split(c)
                        a = float(a)
                        b = float(b)
                if len(distribution) == 2:
                    test_value = distribution[1]
            else:
                raise ValueError("distribution {} not understood".format(distribution))
        else:
            name = "d"
            a = distribution[0]

    elif isinstance(distribution, str):
        if np.any([c in distribution for c in plusminus_chars]):
            name = "n"
            for c in plusminus_chars:
                if c in distribution:
                    a, b = distribution.split(c)
                    a = float(a)
                    b = float(b)
        else:
            raise ValueError("distribution {} not understood".format(distribution))

    elif isinstance(distribution, (int, float, list, np.ndarray, tt.TensorVariable)):
        name = "d"
        a = distribution

    else:
        raise ValueError("distribution {} not understood".format(distribution))

    if a is not None:
        a = method(a)
    if b is not None:
        b = method(b)
    if test_value is not None:
        test_value = method(test_value)

    return [name, a, b, test_value]


def convert_to_pymc3(name, distribution):
    if distribution is None:
        return None
    else:
        distribution_name, a, b, test_value = convert_distribution(distribution)

        if distribution_name == "u":
            return pm.Uniform(name, lower=a, upper=b, testval=test_value)
        elif distribution_name == "n":
            return pm.Normal(name, mu=a, sigma=b, testval=test_value)
        elif distribution_name == "d":
            if isinstance(a, (int, float, list, np.ndarray)):
                return pm.Deterministic(name, tt.as_tensor_variable(a))
            elif isinstance(a, tt.TensorVariable):
                return pm.Deterministic(name, a)
        else:
            raise ValueError("distribution type {} not understood".format(name))
