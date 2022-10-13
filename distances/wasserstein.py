from rpy2 import robjects
from rpy2.robjects import numpy2ri

r = robjects.r
x = r['source']('distances/wasserstein.R')
numpy2ri.activate()

r_wasserstein_dist = robjects.r['WassersteinTest']


def wassertein_distance(sample, dist):
    try:
        return abs(r_wasserstein_dist(sample, dist)[0])
    except:
        print("EXIT EXTI")
        exit()


if __name__ == '__main__':
    print(r_wasserstein_dist)