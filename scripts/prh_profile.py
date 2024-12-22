import sys
import cProfile
import pstats
import numpy as np
sys.path.append("../")
import hill_test
import dghf

def run_many_times(repeats,x_y_kw,bounds):
    for _ in range(repeats):
        for x,y,_ in x_y_kw:
            dghf.fit(x=x, y=y, bounds=bounds)

def run():
    simulated_data = hill_test.MyTestCase().simulated_data
    # ignore the all nan data set
    x_y_kw = [s[0] for s in simulated_data if
              not set(s[0][-1].values()) == set([np.nan])]
    # all of the simulated data has positive hill coefficient
    bounds = [[None, None], [None, None], [None, None], [0, np.inf]]
    profiler = cProfile.Profile()
    profiler.enable()
    run_many_times(repeats=10,x_y_kw=x_y_kw,bounds=bounds)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.dump_stats('program.prof')



if __name__ == "__main__":
    run()
