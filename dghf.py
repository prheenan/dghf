"""
Hill fitting module TBD
"""
import warnings
import numpy as np
import numba
from scipy.optimize import brute, minimize

# float vector for numba
float_vector = numba.types.Array(dtype=numba.float64, ndim=1, layout="C")
int_vector = numba.types.Array(dtype=numba.int64, ndim=1, layout="C")


def get_fit_args(x,y):
    """

    :param x: x values
    :param y: y values
    :return: fit arguments, tuple of (non zero x, y at non zero x, y at zero x )
    """
    idx_zero = np.where(x == 0)[0]
    idx_non_zero = np.where(x != 0)[0]
    x_non_zero = x[idx_non_zero]
    y_non_zero = y[idx_non_zero]
    y_at_x_zero = y[idx_zero]
    return (x_non_zero,y_non_zero,y_at_x_zero)

def hill_log_Ka_hessian(min_v,max_v,log_K_a,n,x):
    """
    see: Wolframalpha output of

    hessian of (v0 + (v1 - v0)/( (k/x)^n +1)) with respect to (v0,v1,k,n)

    then Plain Text -> "Wolfram Language plain text output", and put the string into a variable a

    a.replace("Log","np.log").replace("[","(").replace("]",")").replace("^","**").replace("{","[").replace("}","]")

    :param min_v:
    :param max_v:
    :param log_K_a:
    :param n:
    :param x:
    :return:
    """
    k = np.exp(log_K_a)
    v0 = min_v
    v1 = max_v
    r2 = (1 + (k/x)**n)**2
    r3 = (1 + (k/x)**n)**3
    rn = (n*(k/x)**(-1 + n))
    logr = np.log(k/x)
    f1 = \
        (n * (-v0 + v1) * logr * (k/x)**(-1 + n))/(x*r2) + (2 * n * (-v0 + v1) * (k/x)**(-1 + 2*n) * logr)/(x*r3)
    return \
    [
    [
        0,
        0,
        rn/(x*r2),
        (logr*(k/x)**n)/r2,
    ],
    [
        0,
        0,
        -(rn/(x*r2)),
        -(logr*((k/x)**n )/r2)
    ],
    [
        rn/(x*r2),
        -(rn/(x*r2)),
        -(((-1 + n) * n * (-v0 + v1) (k/x)**(-2 + n))/(r2*(x**2) )) + (2 *(n**2)* (-v0 + v1)* (k/x)**(-2 + 2*n))/(r3*(x**2) ),
        -(((-v0 + v1) *(k/x)**(-1 + n))/(x*r2)) - f1
    ],
    [
        (logr*(k/x)**n )/r2,
        -((logr*(k/x)**n )/r2),
        -(((-v0 + v1) *(k/x)**n)/(k * r2)) - f1,
        -(((-v0 + v1) * logr**2 * (k/x)**n )/r2) + (2 * (-v0 + v1) * (k/x)**(2 * n) * logr**2)/r3
    ]]


def hill_log_Ka_jacobian(x,min_v,max_v,log_K_a,n):
    """

    :param x: see hill_log_Ka
    :param min_v: see hill_log_Ka
    :param max_v:  see hill_log_Ka
    :param log_K_a: see hill_log_Ka
    :param n: see hill_log_Ka
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


@numba.jit(float_vector(float_vector,numba.float64,numba.float64,numba.float64,numba.float64))
def hill_log_Ka(x,min_v,max_v,log_K_a,n):
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
    return min_v + (max_v - min_v )/( (np.exp(log_K_a)/x)**n + 1)

@numba.jit(numba.float64(float_vector,float_vector,float_vector,numba.float64,numba.float64,numba.float64,numba.float64))
def hill_cost(x,y,y_at_x_zero,min_v,max_v,log_K_a,n):
    """

    :param x: x value, should not have any zeros
    :param y:  y value
    "param y_at_x_zero: y values where x is zero
    :param kwargs:  see hill_log_Ka
    :return: sum of squared errors
    """
    # where x is zero, cost is just
    # sum((y_at_x_zero - min_v))*2)
    # since the hill function is just min val at zero concentration
    return sum((y-hill_log_Ka(x=x,min_v=min_v,max_v=max_v,log_K_a=log_K_a,n=n))**2) + \
          (sum(y_at_x_zero-min_v)**2 if y_at_x_zero.size > 0 else 0)

#@numba.jit(float_vector(int_vector, float_vector,float_vector))
def _get_params(variable_index,arg_list,args):
    """

    :param args: from fitter
    :return: dictionary of all fit parameters
    """
    for i, v in zip(variable_index, args):
        arg_list[i] = v
    return arg_list

class Fitter():
    """
    Class to enable fitting of one or more hill parameters
    """
    def __init__(self,param_names,ranges=None,n_brute=0,fixed_params=None):
        """

        :param param_names: names of the paramenters, length N
        :param ranges: list of size N, each element a min and max range
        :param Ns:  number of points to check on coarse grid
        :param fixed_params:  number of fixed parmaeters
        """
        fixed_params = {} if fixed_params is None else fixed_params
        ranges = [] if ranges is None else ranges
        all_names = param_names_order()
        assert set(list(fixed_params.keys())) | set(param_names)  == set(all_names)
        self.arg_list = np.array([ np.nan for _ in all_names],dtype=np.float64)
        for k,v in fixed_params.items():
            self.arg_list[all_names.index(k)] = v
        self.variable_index = np.array([ all_names.index(p) for p in param_names],
                                       dtype=np.int64)
        self.param_names = param_names
        self.ranges = ranges
        self.fixed_params = fixed_params
        self.n_brute = n_brute
        self.opt_dict = {}



    def __call__(self,args,x,y,y_at_x_zero):
        """

        :param args: parameters to fit
        :param x: x values
        :param y: y values
        :param y_at_x_zero: y values where x is zero
        :return: cost associated with these parameters (SSQ)
        """
        return hill_cost(x,y,y_at_x_zero,
                         *_get_params(self.variable_index,self.arg_list,args))

    def fit(self,x,y):
        """

        :param x: x values; concentration
        :param y: y values; signal
        :return: optimized fit
        """
        with warnings.catch_warnings(category=RuntimeWarning):
            warnings.simplefilter("ignore")
            opt = brute(func=self, ranges=self.ranges, args=get_fit_args(x,y),
                        Ns=self.n_brute,full_output=False)
        self.opt_dict = dict([ [k,v]
                               for k,v in zip(self.param_names,opt)])
        for k,v in self.fixed_params.items():
            self.opt_dict[k] = v
        return self.opt_dict


    def jacobian(self,args,x,y,y_zero):
        """

        :param args:  see __call__
        :param x:  see __call__
        :param y:  see __call__
        :return: jacobian given these parameters
        """
        params = _get_params(self.variable_index,self.arg_list,args)
        # cost = sum( y - hill(params) )**2
        # cost jacboian =
        # = sum( -2 (y-hill(params) * jacobian(hill))
        # at x=0, then jacobian is just min_v which is params[0]
        hill = hill_log_Ka(x,*params)
        to_ret = np.sum( -2 * (y-hill) * hill_log_Ka_jacobian(x,*params),axis=1) + \
            (sum(-2 * (y_zero - params[0])) if y_zero.size > 0 else 0)
        return to_ret

    def hessian(self,args,x,y,_):
        """

        :param args: list of argumnets to fit
        :param x:  concentraiton
        :param y:  signal
        :param _:  additional args
        :return: hessian
        """
        params = _get_params(self.variable_index,self.arg_list,args)
        hill = hill_log_Ka(x,*params)
        # loss function is ( y - hill) ^2
        # first deriv wrt x_i is
        # -2 * (y-hill) * dhill/dxi (i.e., last term is jacobian)
        # second deriv wrt xj (giving hessian) is
        # -2 * (y-hill) * ddhill/(dxi*dxj) +  -2 * dhill/dxj * dhill/dxi
        # note for x = 0 the hessian is zero everywhere, so the first term like ddhill/(dxi*dxj) would be zero
        # the second term (dhill/dxj * dhill/dxi) is just the outer product of the jacobian
        jac = hill_log_Ka_jacobian(x,*params)
        return -2 * (y-hill) * hill_log_Ka_hessian(x,*params) - 2 * np.outer(jac,jac)

def _get_ranges(x,y,range_val_initial=None,range_conc_initial=None,
                range_n_intial=None,bounds=None):
    """

    :param x: concentration
    :param y:  signal
    :param range_val_initial: 2-tuple; min and max of coarse grid for y values
    :param range_conc_initial: 2-tuple, min and max of corase grid for log scaled concentration
    :param range_n_intial:  2-tuple, min and max of coarse grid for hill coefficient
    :param bounds: final bounds for fit
    :return:
    """
    idx_non_zero = np.where(x != 0)[0]
    conc_min, conc_max = min(np.log(x[idx_non_zero])), max(np.log(x[idx_non_zero]))
    min_v, max_v = min(y), max(y)
    dv = min_v + max_v
    if range_val_initial is None:
        range_val_initial = [min_v - dv / 2, max_v + dv / 2]
    if range_conc_initial is None:
        range_conc_initial = [conc_min - np.log(3), conc_max + np.log(3)]
    if range_n_intial is None:
        range_n_intial = [-5, 5]
    ranges = [range_val_initial, range_val_initial, range_conc_initial, range_n_intial]
    if bounds is not None:
        # modify the ranges according to any initial constraints the user gives as bounds
        ranges_final = []
        for range_v, bounds_v in zip(ranges, bounds):
            range_i = []
            for r1, b1, func in zip(range_v, bounds_v, [max, min]):
                range_i.append(r1 if b1 is None else func(r1, b1))
            ranges_final.append(range_i)
    else:
        ranges_final = ranges
    return ranges_final

def param_names_order():
    """

    :return: in-order parameter names
    """
    return ['min_v', 'max_v', 'log_K_a', 'n']

def _initial_guess(x,y,ranges,coarse_n,fine_n):
    """

    :param x: x values (concentrations)
    :param y:  signals
    :param ranges: list of ranges
    :param coarse_n: number of coarse grid for brute
    :param fine_n:  number of fine grid for brute
    :return: initial dictionary of guesses
    """
    _, _, range_conc_initial, _ = ranges
    all_names = param_names_order()
    fit_all = Fitter(param_names=all_names, ranges=ranges, n_brute=coarse_n)
    fit_all.fit(x=x, y=y)
    fixed_params = { k:v for k, v in fit_all.opt_dict.items()
                     if k != "log_K_a"}
    fit_Ka = Fitter(param_names=['log_K_a'], n_brute=fine_n,
                    ranges=[range_conc_initial],
                    fixed_params=fixed_params)
    fit_Ka.fit(x=x, y=y)
    # use the coarse grid for all the other parameters
    p0 = fit_all.opt_dict
    # overwrite
    p0["log_K_a"] = fit_Ka.opt_dict["log_K_a"]
    # final fitter has no fixed parameters; completely free
    x0 = [p0[n] for n in all_names]
    return x0

def fit(x,y,coarse_n=7,fine_n=1000,bounds=None,method='L-BFGS-B',**kw):
    """

    :param x: concentration , length
    :param y: signal, length N
    :param coarse_n:  number of parameters for coare parameter estimation
    :param fine_n:  numbr of parameters for potency estimation
    :param kw: bounds on the initial guess
    :param bounds: bounds on the final fit
    :return: dictionary of best fit parameters for hill_log_Ka
    """
    # convert to float64 (numpy specifically requires)
    x = np.array(x,dtype=np.float64)
    y = np.array(y,dtype=np.float64)
    idx_not_nan = np.where(~(np.isnan(x) | np.isnan(y)))[0]
    if len(idx_not_nan) <= 2:
        warnings.warn("Need at least 4 non-nan points;"+\
                      f"only had {len(idx_not_nan)}. Fit values undefined (nan)")
        return dict(min_v=np.nan, max_v=np.nan, log_K_a=np.nan, n=np.nan)
    x = x[idx_not_nan]
    y = y[idx_not_nan]
    ranges = _get_ranges(x=x,y=y,bounds=bounds,**kw)
    x0 = _initial_guess(x=x,y=y,ranges=ranges,coarse_n=coarse_n,fine_n=fine_n)
    fit_final = Fitter(param_names=param_names_order())
    with warnings.catch_warnings(category=RuntimeWarning):
        warnings.simplefilter("ignore")
        minimize_v = minimize(fun=fit_final, args=get_fit_args(x,y),
                              method=method,
                              jac=fit_final.jacobian, x0=x0,bounds=bounds)
    # fit everything in a free manner
    kw_fit = dict([[n, x_i] for n, x_i in zip(fit_final.param_names, minimize_v.x)])
    return kw_fit
