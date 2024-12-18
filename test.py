"""
Testing module for hill fitter
"""
import unittest
import numpy as np
from scipy.optimize import brute, minimize
import plotly.graph_objects as go


def hill_log_Ka_jacobian(min_v,max_v,log_K_a,n,x):
    """

    :param min_v: see hill_log_Ka
    :param max_v:  see hill_log_Ka
    :param log_K_a: see hill_log_Ka
    :param n: see hill_log_Ka
    :param x: see hill_log_Ka
    :return: jacobian of the hill equation
    """
    ratio = np.exp(log_K_a)/x
    exp = ratio**n
    denom = (exp + 1)**-1
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


class Fitter():
    def __init__(self,param_names,ranges=None,n_brute=0,fixed_params=None):
        """

        :param param_names: names of the paramenters, length N
        :param ranges: list of size N, each element a min and max range
        :param Ns:  number of points to check on coarse grid
        :param fixed_params:  number of fixed parmaeters
        """
        fixed_params = {} if fixed_params is None else fixed_params
        ranges = [] if ranges is None else ranges
        self.param_names = param_names
        self.ranges = ranges
        self.fixed_params = fixed_params
        self.n_brute = n_brute
        self.opt_dict = {}

    def _get_params(self,args):
        """

        :param args: from fitter
        :return: dictionary of all fit parameters
        """
        kw = dict([ [k,v] for k,v in zip(self.param_names,args)])
        for k,v in self.fixed_params.items():
            assert k not in kw
            kw[k] = v
        return kw

    def __call__(self,args,x,y):
        """

        :param args: parameters to fit
        :param x: x values
        :param y: y values
        :return: cost associated with these parameters (SSQ)
        """
        kw = self._get_params(args=args)
        return hill_cost(x=x,y=y,**kw)

    def fit(self,x,y):
        """

        :param x: x values; concentration
        :param y: y values; signal
        :return: optimized fit
        """
        opt = brute(func=self, ranges=self.ranges, args=(x,y), Ns=self.n_brute,
                    full_output=False)
        self.opt_dict = dict([ [k,v]
                               for k,v in zip(self.param_names,opt)])
        for k,v in self.fixed_params.items():
            self.opt_dict[k] = v
        return self.opt_dict

    def jacobian(self,args,x,y):
        """

        :param args:  see __call__
        :param x:  see __call__
        :param y:  see __call__
        :return: jacobian given these parameters
        """
        kw = self._get_params(args=args)
        # cost = sum( y - hill(params) )**2
        # cost jacboian =
        # = sum( -2 (y-hill(params) * jacobian(hill))
        hill = hill_log_Ka(x=x,**kw)
        to_ret = np.sum( -2 * (y-hill) * hill_log_Ka_jacobian(x=x,**kw),axis=1)
        return to_ret

def fit(x,y,coarse_n=10,fine_n=1000,
        range_val_initial=None,range_conc_initial=None,
        range_n_intial=None):
    """

    :param x: concentration , length
    :param y: signal, length N
    :param coarse_n:  number of parameters for coare parameter estimation
    :param fine_n:  numbr of parameters for potency estimation
    :param range_val_initial: 2-tuple; min and max of coarse grid for y values
    :param range_conc_initial: 2-tuple, min and max of corase grid for log scaled concentration
    :param range_n_intial:  2-tuple, min and max of coarse grid for hill coefficient
    :return: dictionary of best fit parameters for hill_log_Ka
    """
    conc_min, conc_max = min(np.log(x)), max(np.log(x))
    min_v, max_v = min(y), max(y)
    dv = min_v + max_v
    if range_val_initial is None:
        range_val_initial = [min_v - dv / 2, max_v + dv / 2]
    if range_conc_initial is None:
        range_conc_initial = [conc_min - np.log(3), conc_max + np.log(3)]
    if range_n_intial is None:
        range_n_intial = [0, 5]
    ranges = [range_val_initial, range_val_initial, range_conc_initial, range_n_intial]
    all_names = ['min_v', 'max_v', 'log_K_a', 'n']
    fit_all = Fitter(param_names=all_names, ranges=ranges, n_brute=coarse_n)
    fit_all.fit(x=x, y=y)
    fit_Ka = Fitter(param_names=['log_K_a'], n_brute=fine_n,
                    ranges=[range_conc_initial],
                    fixed_params=dict(
                        [[k, v] for k, v in fit_all.opt_dict.items()
                         if k != "log_K_a"]))
    fit_Ka.fit(x=x, y=y)
    # use the coarse grid for all the other parameters
    p0 = fit_all.opt_dict
    # overwrite
    p0["log_K_a"] = fit_Ka.opt_dict["log_K_a"]
    # final fitter has no fixed parameters; completely free
    fit_final = Fitter(param_names=['min_v', 'max_v', 'log_K_a', 'n'])
    x0 = [p0[n] for n in fit_final.param_names]
    minimize_v = minimize(fun=fit_final, args=(x, y), jac=fit_final.jacobian, x0=x0)
    # fit everything in a free manner
    kw_fit = dict([[n, x_i] for n, x_i in zip(fit_final.param_names, minimize_v.x)])
    return kw_fit

def _get_x_y_k(log10_i=-9,log10_f=-4,noise_scale=0,**kw):
    """

    :param log10_i: which magnitude to start (e.g., -6 is 1 uM
    :param log10_f:  which magnitude to end  (e.g., -3 is 1 mM)
    :param noise_scale: scale of noise
    :param kw: see hill_log_Ka
    :return: tuple of (x values, y values, kw)
    """
    x = np.logspace(log10_i, log10_f, num=11,base=10)
    y = hill_log_Ka(x=x, **kw) + \
        np.random.normal(loc=0,scale=noise_scale,size=x.size)
    return x,y,kw


class MyTestCase(unittest.TestCase):

    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        self.i_subtest = 0
        np.random.seed(42)

    def _assert_close_kw(self,kw_found,kw_expected,atol=0,rtol=1e-6,
                         atol_signal=None,rtol_signal=None):
        """

        :param kw_found: fitted keyworkds
        :param kw_expected:  what we expect
        :param atol: input to np.testing.assert_allclose for hill/logKa
        :param rtol: input to np.testing.assert_allclose for hill/logKa
        :param atol_signal: input to np.testing.assert_allclose for min/max
        :param rtol_signal: input to np.testing.assert_allclose for min/max
        :return: asserts true/value if available
        """
        if atol_signal is None:
            atol_signal = atol
        if rtol_signal is None:
            rtol_signal = rtol
        signal_names = ['min_v', 'max_v',]
        non_signal_names = ['log_K_a', 'n']
        for names,atol_i,rtol_i in [ [non_signal_names,atol,rtol],
                                     [signal_names,atol_signal,rtol_signal]]:
            fit_found = [kw_found[n] for n in names]
            fit_expected = [ kw_expected[n] for n in names]
            with self.subTest(self.i_subtest):
                np.testing.assert_allclose(fit_found,fit_expected, atol=atol_i,
                                           rtol=rtol_i)
                self.i_subtest += 1


    def test_something(self):
        kw_fit_1 = dict(min_v=10, max_v=100, log_K_a=np.log(1e-6),n=1)
        x_y_k_err = [
            [_get_x_y_k(-7, -4, **kw_fit_1),dict(atol=1e-4)],
            [_get_x_y_k(-7, -4, noise_scale=5,**kw_fit_1), dict(rtol=0.15,
                                                                atol_signal=5)],
        ]
        for (x,y,kw_expected),kw_err in x_y_k_err[::-1]:
            kw_fit = fit(x,y)
            self._assert_close_kw(kw_found=kw_fit, kw_expected=kw_expected,
                                  **kw_err)


def _debug_plot(x,y,kw_fit):
    """

    :param x: x values
    :param y: y values
    :param kw_fit:  fitted x and y
    :return:  nothing, gives plot
    """
    x_interp = np.logspace(min(np.log10(x / 10)), max(np.log10(x * 10)), base=10)
    y_interp = hill_log_Ka(x=x_interp, **kw_fit)
    fig = go.Figure()
    # Add the first line with markers
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name="Data"))
    # Add the second line without markers
    fig.add_trace(go.Scatter(x=x_interp, y=y_interp, mode='lines', name="Fit"))
    fig.update_layout(xaxis_type="log")
    fig.show()


if __name__ == '__main__':
    unittest.main()
