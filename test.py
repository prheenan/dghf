import unittest
import numpy as np
from plotly import express as px
import plotly.graph_objects as go
from scipy.optimize import brute, minimize
import dhgf

def hill_log_Ka_jacobian(min_v,max_v,log_K_a,n,x):
    """

    :param min_v: see hill_log_Ka
    :param max_v:  see hill_log_Ka
    :param log_K_a: see hill_log_Ka
    :param n: see hill_log_Ka
    :param x: see hill_log_Ka
    :return: jacobian of the hill equation
    """
    ratio = (np.exp(log_K_a)/x)
    exp = ratio**n
    denom = ( exp + 1)**-1
    return np.array([
        1 - denom,
        denom,
        (max_v - min_v) * (-n * exp)* (denom**2),
        (max_v - min_v) * (-exp * np.log(ratio)) * denom**2,
    ])

def hill_log_Ka(min_v,max_v,log_K_a,n,x):
    """
    See https://reaction-networks.net/wiki/Hill_kinetics
    theta = ( (K_a/L)^n + 1)-1
    theta = ( (exp(log_K_a)/L)^n + 1)-1

    :param x:  concentration, units of exp(log_K_a)
    :param min_v:  minimum value
    :param max_v:  maximum value
    :param log_K_a: log of concentration
    :param n: hill fit coefficient
    :return: hill fit information
    """
    return min_v + (max_v - min_v ) * ( (np.exp(log_K_a)/x)**n + 1)**-1

def hill_cost(x,y,**kwargs):
    """

    :param x: x value
    :param y:  y value
    :param kwargs:  see hill_log_Ka
    :return: sum of squared errors
    """
    return sum((y-hill_log_Ka(x=x,**kwargs))**2)


class Fitter(object):
    def __init__(self,param_names,ranges=list(),Ns=0,fixed_params=dict()):
        self.param_names = param_names
        self.ranges = ranges
        self.fixed_params = fixed_params
        self.Ns= Ns

    def _get_params(self,args):
        kw = dict([ [k,v] for k,v in zip(self.param_names,args)])
        for k,v in self.fixed_params.items():
            assert k not in kw
            kw[k] = v
        return kw

    def __call__(self,args,x,y):
        kw = self._get_params(args=args)
        return hill_cost(x=x,y=y,**kw)

    def fit(self,x,y):
        self._opt = brute(func=self, ranges=self.ranges, args=(x,y), Ns=self.Ns,
                          full_output=False)
        self.opt_dict = dict([ [k,v]
                               for k,v in zip(self.param_names,self._opt)])
        for k,v in self.fixed_params.items():
            self.opt_dict[k] = v
        return self.opt_dict

    def jacobian(self,args,x,y):
        kw = self._get_params(args=args)
        # cost = sum( y - hill(params) )**2
        # cost jacboian =
        # = sum( -2 (y-hill(params) * jacobian(hill))
        hill = hill_log_Ka(x=x,**kw)
        to_ret = np.sum( -2 * (y-hill) * hill_log_Ka_jacobian(x=x,**kw),axis=1)
        return to_ret


class MyTestCase(unittest.TestCase):
    def test_something(self):
        x = np.logspace(-9,-4,num=11)
        y = hill_log_Ka(x=x, min_v=0, max_v=100, log_K_a=np.log(1e-6), n=1)
        conc_min, conc_max = min(np.log(x)), max(np.log(x))
        min_v , max_v = min(y), max(y)
        dv = min_v + max_v
        range_val = [min_v-dv/2,max_v+dv/2]
        range_conc = [conc_min-np.log(3),conc_max+np.log(3)]
        ranges = [ range_val, range_val, range_conc, [-5,5]]
        Ns = 10
        fit_all = Fitter(param_names=['min_v','max_v','log_K_a','n'],
                         ranges=ranges,Ns=Ns)
        fit_all.fit(x=x,y=y)
        fit_Ka = Fitter(param_names=['log_K_a'],Ns=1000,
                        ranges=[range_conc],
                        fixed_params=dict([ [k,v] for k,v in fit_all.opt_dict.items()
                                            if k != "log_K_a"]))
        fit_Ka.fit(x=x,y=y)
        # use the coarse grid for all the other parameters
        p0 = fit_all.opt_dict
        # overwrite
        p0["log_K_a"] = fit_Ka.opt_dict["log_K_a"]
        # final fitter has no fixed parameters; completely free
        fit = Fitter(param_names=['min_v','max_v','log_K_a','n'])
        x0 = [p0[n] for n in fit.param_names]
        minimize_v = minimize(fun=fit,args=(x,y),jac=fit.jacobian,
                              x0=x0)
        # fit everything in a free manner
        kw_fit = dict([ [n,x_i] for n,x_i in zip(fit.param_names,minimize_v.x)])
        x_interp = np.logspace(-10,-3,num=200)
        y_interp = hill_log_Ka(x=x_interp,**kw_fit)
        fig = go.Figure()
        # Add the first line with markers
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',name="Data"))
        # Add the second line without markers
        fig.add_trace(go.Scatter(x=x_interp, y=y_interp, mode='lines',name="Fit"))
        fig.update_layout(xaxis_type="log")
        fig.show()


if __name__ == '__main__':
    unittest.main()
