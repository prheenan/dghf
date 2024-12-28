"""
Hill fitting module TBD
"""
import warnings
import json
from multiprocessing import Pool, cpu_count
import numpy as np
import numba
import pandas
from tqdm import tqdm
from scipy.optimize import brute, minimize, fmin_powell, fmin, fmin_cg, fmin_bfgs
from matplotlib import pyplot as plt
import click
from click import ParamType
# float vector for numba
float_vector = numba.types.Array(dtype=numba.float64, ndim=1, layout="C")
from scripts import canvass_download


@click.group()
def cli():
    """
    Defines the click command line interface; nothing to do
    """

class FinishType(ParamType):
    """
    For conversion from string to actual brute finish function
    """
    def __init__(self):
        """
        Initialize; nothing to do
        """

    def get_metavar(self, _):
        """
        :return: help string for this
        """
        return 'Choice([fmin_powell, fmin, fmin_cg, fmin_bfgs])'

    def convert(self, value, _, __):
        """

        :param value: string given by get_metavar
        :return: actual function to use
        """
        convert_dict = {'fmin_powell':fmin_powell,
                        'fmin':fmin,
                        'fmin_cg':fmin_cg,
                        'fmin_bfgs':fmin_bfgs}
        lower = str(value).lower()
        if lower in convert_dict:
            return convert_dict[lower]
        else:
            self.fail(f"Invalid value: {value}. Expected {self.get_metavar(None)}")

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
    denom = 1/(exp + 1)
    return np.array([
        1 - denom,
        denom,
        (max_v - min_v) * (-n * exp)* (denom**2),
        (max_v - min_v) * (-exp * np.log(ratio)) * denom**2,
    ])


@numba.njit(float_vector(numba.float64,numba.float64,numba.float64,numba.float64,float_vector))
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
    return min_v + (max_v - min_v )/( (np.exp(log_K_a)/x)**n + 1)

@numba.njit(numba.float64(float_vector,float_vector,float_vector,float_vector))
def hill_cost(args,x,y,y_at_x_zero):
    """

    :param x: x value, should not have any zeros
    :param y:  y value
    "param y_at_x_zero: y values where x is zero
    :param kwargs:  see hill_log_Ka
    :return: sum of squared errors
    """
    min_v, max_v, log_K_a, n = args
    # where x is zero, cost is just
    # sum((y_at_x_zero - min_v))*2)
    # since the hill function is just min val at zero concentration
    return sum((y-hill_log_Ka(min_v,max_v,log_K_a,n,x))**2) + \
          (sum(y_at_x_zero-min_v)**2 if y_at_x_zero.size > 0 else 0)


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
        # all parameters should be accounted for
        assert set(fixed_params.keys()) | set(param_names) == set(param_names_order())
        # fixed parameters and parameters to fit shouldn't overlap
        assert set(fixed_params.keys()) & set(param_names) == set()
        self.param_names = param_names
        self.ranges = ranges
        self.fixed_params = fixed_params
        self.n_brute = n_brute
        self.opt_dict = {}
        self.fixed = np.array([fixed_params[p] if p in fixed_params else np.nan
                               for p in param_names_order() ],dtype=np.float64)
        if param_names == param_names_order():
            # fitting everything
            self._f_hill_cost = hill_cost
        elif param_names == ["log_K_a"]:
            # fitting only logKa
            self._f_hill_cost = self._log_Ka_hill_cost
        else:
            raise ValueError("Not supported")

    def _log_Ka_hill_cost(self,args,x,y,y_at_x_zero):
        """

        :param args:
        :param x:
        :param y:
        :param y_at_x_zero:
        :return:
        """
        self.fixed[2] = args[0]
        return hill_cost(self.fixed,x, y, y_at_x_zero)


    def _get_params(self,args):
        """

        :param args: from fitter
        :return: dictionary of all fit parameters
        """
        return { k:v for k,v in zip(self.param_names,args) } | self.fixed_params


    def fit(self,x,y,**kw):
        """

        :param x: x values; concentration
        :param y: y values; signal
        :return: optimized fit
        """
        with warnings.catch_warnings(category=RuntimeWarning):
            warnings.simplefilter("ignore")
            opt = brute(func=self._f_hill_cost, ranges=self.ranges,
                        args=get_fit_args(x,y),
                        Ns=self.n_brute,full_output=False,**kw)
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
        kw = self._get_params(args=args)
        # cost = sum( y - hill(params) )**2
        # cost jacboian =
        # = sum( -2 (y-hill(params) * jacobian(hill))
        hill = hill_log_Ka(x=x,**kw)
        to_ret = np.sum( -2 * (y-hill) * hill_log_Ka_jacobian(x=x,**kw),axis=1) + \
            (sum(-2 * (y_zero - kw["min_v"])) if y_zero.size > 0 else 0)
        return to_ret

    def hessian(self,args,x,y,_):
        """

        :param args: list of argumnets to fit
        :param x:  concentraiton
        :param y:  signal
        :param _:  additional args
        :return: hessian
        """
        kw = self._get_params(args=args)
        hill = hill_log_Ka(x=x,**kw)
        # loss function is ( y - hill) ^2
        # first deriv wrt x_i is
        # -2 * (y-hill) * dhill/dxi (i.e., last term is jacobian)
        # second deriv wrt xj (giving hessian) is
        # -2 * (y-hill) * ddhill/(dxi*dxj) +  -2 * dhill/dxj * dhill/dxi
        # note for x = 0 the hessian is zero everywhere, so the first term like ddhill/(dxi*dxj) would be zero
        # the second term (dhill/dxj * dhill/dxi) is just the outer product of the jacobian
        jac = hill_log_Ka_jacobian(x=x,**kw)
        return -2 * (y-hill) * hill_log_Ka_hessian(x=x,**kw) - 2 * np.outer(jac,jac)

def _harmonize_range(range_v, bounds_v):
    range_i = []
    for r1, b1, func in zip(range_v, bounds_v, [max, min]):
        # if the bounds is inf or None, then stick with the original
        range_i.append(r1 if b1 is None or np.isinf(b1) or np.isnan(b1)
                       else func(r1, b1))
    return range_i

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
        range_conc_initial = [conc_min, conc_max]
    if range_n_intial is None:
        range_n_intial = [-5, 5]
    ranges = [range_val_initial, range_val_initial, range_conc_initial, range_n_intial]
    if bounds is not None:
        # modify the ranges according to any initial constraints the user gives as bounds
        ranges_final = [_harmonize_range(range_v, bounds_v)
                        for range_v, bounds_v in zip(ranges, bounds)]
    else:
        ranges_final = ranges
    return ranges_final

def param_names_order():
    """
    :return: in-order parameter names
    """
    return ['min_v', 'max_v', 'log_K_a', 'n']

def _initial_guess(x,y,ranges,coarse_n,fine_n,
                   finish=fmin_powell):
    """

    :param x: x values (concentrations)
    :param y:  signals
    :param ranges: list of ranges
    :param coarse_n: number of coarse grid for brute
    :param fine_n:  number of fine grid for brute
    :param finish: what function to finish brute using
    :return: initial dictionary of guesses
    """
    _, _, range_conc_initial, _ = ranges
    all_names = param_names_order()
    fit_all = Fitter(param_names=all_names, ranges=ranges, n_brute=coarse_n)
    fit_all.fit(x=x, y=y,finish=finish)
    fixed_params = {k: v for k, v in fit_all.opt_dict.items()
                    if k != "log_K_a"}
    fit_Ka = Fitter(param_names=['log_K_a'], n_brute=fine_n,
                    ranges=[range_conc_initial],
                    fixed_params=fixed_params)
    fit_Ka.fit(x=x, y=y,finish=finish)
    # use the coarse grid for all the other parameters
    p0 = fit_all.opt_dict
    # overwrite
    p0["log_K_a"] = fit_Ka.opt_dict["log_K_a"]
    # final fitter has no fixed parameters; completely free
    x0 = np.array([p0[n] for n in all_names],dtype=np.float64)
    return x0

def _fit_multiprocess(kw):
    """

    :param kw:  see fit
    :return:  see fit
    """
    return fit(**kw)

def set_bounds_if_inactive(inactive_range,x,y,
                           bounds_min_v,bounds_max_v,bounds_log_K_a,bounds_n):
    """
    Modify the bounds if all of the data are inactive or active

    :param inactive_range: tuple of bounds; points within this bound are considered inactive
    :param x: concentration
    :param y: signal
    :param bounds_min_v: user specified bounds; if more stringent will ovewrite what we find here
    :param bounds_max_v: user specified bounds; if more stringent will ovewrite what we find here
    :param bounds_log_K_a: user specified bounds; if more stringent will ovewrite what we find here
    :param bounds_n: user specified bounds; if more stringent will ovewrite what we find here
    :return: tuple of updated [bounds_min_v,bounds_max_v,bounds_log_K_a,bounds_n]
    """
    range_defined = inactive_range is not None and \
                    inactive_range[0] is not None and \
                    inactive_range[1] is not None
    if (range_defined and any(x > 0)):
        idx_x = np.where(x > 0)[0]
        x_non_zero = x[idx_x]
        min_r, max_r = min(inactive_range), max(inactive_range)
        point_inactive = (y <= max_r) & (y >= min_r)
        triggered = False
        if all(point_inactive | np.isnan(y)):
            # then the log Ka must be set to the maximum
            bounds_log_K_a = [np.log(max(x_non_zero)),np.log(max(x_non_zero))]
            triggered = True
        elif (not any(point_inactive)) and any(x > 0):
            # no point is inactive; bounds must be set to minimum
            bounds_log_K_a = [np.log(min(x_non_zero)),np.log(min(x_non_zero))]
            triggered = True
        if triggered:
            dy = max(y) - min(y)
            bounds_new = [min(y) - dy, max(y) + dy]
            bounds_min_v = _harmonize_range(bounds_new, bounds_min_v)
            bounds_max_v = _harmonize_range(bounds_new, bounds_max_v)
    return [bounds_min_v,bounds_max_v,bounds_log_K_a,bounds_n]

def fit(x,y,coarse_n=7,fine_n=100,bounds_min_v = None,
        bounds_max_v=None,bounds_log_K_a=None,
        bounds_n=None,method='L-BFGS-B',inactive_range=None,
        finish=fmin_powell,**kw):
    """

    :param x: concentration , length
    :param y: signal, length N
    :param coarse_n:  number of parameters for coare parameter estimation
    :param fine_n:  numbr of parameters for potency estimation
    :param kw: bounds on the initial guess
    :param bounds_minv: boundsfor potency, unbounded is [None,None]
    :param bounds_maxv: bounds for potency, unbounded is [None,None]
    :param bounds_log_Ka: bounds for potency, unbounded is [None,None]
    :param bounds_n: bounds for hill coefficient, unbounded is [None,None]
    :param fmin: minimizatio function after brute
    :param method: method for optimization function
    :param inactive_range: inactive range; for example [-inf,50] would mean
    points are considered inactive if they are between -inf and 50
    :return: dictionary of best fit parameters for hill_log_Ka
    """
    bounds_min_v = [None, None] if bounds_min_v is None else bounds_min_v
    bounds_max_v = [None, None] if bounds_max_v is None else bounds_max_v
    bounds_log_K_a = [None, None] if bounds_log_K_a is None else bounds_log_K_a
    bounds_n = [None, None] if bounds_n is None else bounds_n
    # convert to float64 (numpy specifically requires)
    x = np.array(x,dtype=np.float64)
    y = np.array(y,dtype=np.float64)
    idx_not_nan = np.where(~(np.isnan(x) | np.isnan(y)))[0]
    if len(idx_not_nan) <= 2:
        warnings.warn("Need at least 4 non-nan points;"+\
                      f"only had {len(idx_not_nan)}. Fit values undefined (nan)")
        return {"min_v":np.nan, "max_v":np.nan, "log_K_a":np.nan, "n":np.nan}
    x = x[idx_not_nan]
    y = y[idx_not_nan]
    bounds = set_bounds_if_inactive(inactive_range,x,y,bounds_min_v,
                                    bounds_max_v,bounds_log_K_a,bounds_n)
    ranges = _get_ranges(x=x,y=y,bounds=bounds,**kw)
    x0 = _initial_guess(x=x,y=y,ranges=ranges,coarse_n=coarse_n,fine_n=fine_n,
                        finish=finish)
    fit_final = Fitter(param_names=param_names_order())
    with warnings.catch_warnings(category=RuntimeWarning):
        warnings.simplefilter("ignore")
        minimize_v = minimize(fun=fit_final._f_hill_cost, args=get_fit_args(x,y),
                              method=method,
                              jac=fit_final.jacobian, x0=x0,bounds=bounds)
    # fit everything in a free manner
    kw_fit = dict(zip(fit_final.param_names, minimize_v.x))
    return kw_fit


def gallery_plot(x_y,all_fit_kw,n_rows=None,n_cols=None,
                 ids_plot=None,figsize=(5,4)):
    """

    :param x_y: list of length N; each element tupe of concentration and y values
    :param all_fit_kw: list of length N; each element is output of fit function
    :param n_rows: number of rows; defaults to sqrt of N
    :param n_cols: number of cols; defaults to sqrt of N
    :param ids_plot: to use on each plot title
    :param figsize: size of figure to use
    :return:  matpotlib figure
    """
    show_title_debug = ids_plot is not None
    if n_rows is None:
        n_rows = int(np.ceil(np.sqrt(len(all_fit_kw))))
        n_cols = n_rows
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # the olive color isn't readable
    colors = [c for i,c in enumerate(prop_cycle.by_key()['color'])
              if i != 8]
    plt.close("all")
    plt.style.use("bmh")
    fig, axs = plt.subplots(n_rows,n_cols,figsize=figsize,
                            sharex=True,sharey=True,dpi=100)
    i_plot = 0
    for row in axs:
        for col in row:
            x,y  = x_y[i_plot]
            kw = all_fit_kw[i_plot]
            x_pred = np.logspace(np.log10(min(x)),np.log10(max(x)),
                                 base=10,endpoint=True)
            y_pred = hill_log_Ka(x=x_pred, **kw)
            col.semilogx(x,y,'o',markersize=4,markerfacecolor='w',
                         color=colors[i_plot % len(colors)])
            col.semilogx(x_pred,y_pred,'--',color='k',linewidth=1.5)
            if ids_plot is not None and show_title_debug:
                col.set_title(ids_plot[i_plot],fontsize=6,y=0.7,x=0.2)
            i_plot += 1
    for ax in axs.flat:
        ax.set(xlabel='Cmpd [M]', ylabel='Activity\n(%)')
    for ax in axs.flat:
        ax.label_outer()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    return fig


def _ids_to_xy(in_df,col_x="Concentration (M)",col_y="Activity (%)",
               col_id="Curve ID"):
    """

    :param in_df:  see _fit_df
    :param col_x:  see _fit_df
    :param col_y: see _fit_df
    :param col_id: see _fit_df
    :return: dictionary going from id to x,y assays
    """
    return {id_v: {'x': df_v[col_x].to_numpy(),
                   'y': df_v[col_y].to_numpy()}
            for id_v, df_v in in_df.groupby(col_id)}

def _fit_df(in_df,n_jobs=None,col_x="Concentration (M)",
            col_y="Activity (%)",col_id="Curve ID",**kw):
    """

    :param in_df: data frame
    :param n_jobs:  number of processors
    :param col_x: x column in dataframe to use
    :param col_y:  y olumn in dataframe to use
    :param col_id: id column to use
    :param kw: passed to fit
    :return: tuple of ( list of (x,y) of size N, list of ids of size N, list of fit kws of size N)
    """
    x_y_dict = _ids_to_xy(in_df, col_x, col_y, col_id)
    ids = sorted(set(in_df[col_id]))
    x_y = [x_y_dict[i] | kw for i in ids]
    n_curves = len(x_y)
    kw_tqdm = dict(total=n_curves,desc="Fitting")
    if n_jobs is None or n_jobs in [0,1]:
        all_fit_kw = list(tqdm(map(_fit_multiprocess,x_y),**kw_tqdm))
    else:
        # n_jobs is a number greater than 1 or less than 0
        if n_jobs < 0:
            # use all cpus except n_jobs
            n_jobs = cpu_count() + n_jobs
        with Pool(n_jobs) as p:
            all_fit_kw = list(tqdm(p.imap(_fit_multiprocess, x_y),**kw_tqdm))
    return x_y,ids,all_fit_kw

def _fit_file_helper(input_file,output_file=None,col_id="Curve ID",**kw):
    """

    :param input_file: input file, should be csv
    :param output_file: output file, should be csv or json
    :param col_id: id column in input_file
    :param kw:  see _fit_df
    :return: nothing
    """
    if output_file is None:
        output_file = input_file + "_out.csv"
    # check the file extensions
    if not input_file.endswith('.csv'):
        raise click.BadParameter('Input file must have a .csv extension.')
    if not (output_file.endswith('.csv') | output_file.endswith('.json')):
        raise click.BadParameter('Input file must have a .csv or .json extension.')
    if output_file is None:
        output_file = f"{input_file}_PRH.csv"
    df = pandas.read_csv(input_file)
    _, ids, all_fit_kw = _fit_df(in_df=df,**kw)
    list_of_dicts = [kw | {col_id:id_v} for kw,id_v in zip(all_fit_kw,ids)]
    if output_file.endswith(".csv"):
        pandas.DataFrame(list_of_dicts).to_csv(output_file,index=False)
    elif output_file.endswith(".json"):
        with open(output_file,'w',encoding="utf8") as f:
            json.dump(list_of_dicts,f)


@cli.command()
@click.option('--output_file', required=True,type=click.Path(dir_okay=False),
              help="Name of output file (only csv supported)")
@click.option('--random_sample', required=False,
              default=None,type=int,help="Number of dose response curves to return")
@click.option('--random_seed', required=False,
              default=42,type=int,help="Random seed for sampling")
@click.option('--out_dir', required=False,type=click.Path(dir_okay=True),
              help="Directory to put the individual downloads",
              default="./out/test/cache_canvass")
def export_canvass(output_file,**kw):
    canvass_download.read_canvass_data(**kw).to_csv(output_file,index=False)

@cli.command()
@click.option('--input_file', required=True,type=click.Path(exists=True,dir_okay=False),
              help="Name of input file (must be csv)")
@click.option('--output_file', required=False,type=click.Path(dir_okay=False),
              help="Name of output file (json, csv supported)",default=None)
@click.option('--col_x', required=False,type=str,
              help="x column (e.g., concentation)",default="Concentration (M)")
@click.option('--col_y', required=False,type=str,
              help="y column (e.g., activity)",default="Activity (%)")
@click.option('--col_id', required=False,type=str,
              help="id column (e.g., concentation)",default="Curve ID")
@click.option('--coarse_n', required=False,
              default=7,type=int,help="Number of coarse grid points")
@click.option('--fine_n', required=False,
              default=100,type=int,help="Number of fine grid points for potency")
@click.option('--bounds_min_v',required=False, nargs=2, type=float,
              default=[None,None],help="Bounds for minimum fit value")
@click.option('--bounds_max_v',required=False, nargs=2, type=float,
              default=[None,None],help="Bounds for maximum fit value")
@click.option('--bounds_log_K_a',required=False, nargs=2, type=float,
              default=[None,None],help="Bounds for natural logarithm of Ka")
@click.option('--bounds_n',required=False, nargs=2, type=float,
              default=[None,None],help="Bounds for hill coefficient")
@click.option('--inactive_range',required=False, nargs=2, type=float,
              default=[None,None],help="Bounds for hill coefficient")
@click.option('--method',required=False,  type=str,
              default='L-BFGS-B',help="Bounds for hill coefficient")
@click.option('--finish',required=False,  type=FinishType(),
              default='fmin_powell',help="Function for coarse grid finishing")
@click.option('--n_jobs',required=False,  type=int,
              default=1,help="Number of cores to use; defaults to 1")
def fit_file(**kw):
    """

    :param kw: see  _fit_file_helper
    :return: see _fit_file_helper
    """
    # convert to correct spelling; click forces lowercase
    kw["bounds_log_K_a"] = kw['bounds_log_k_a']
    del kw['bounds_log_k_a']
    return _fit_file_helper(**kw)

if __name__ == '__main__':
    cli()
