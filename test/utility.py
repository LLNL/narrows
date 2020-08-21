import matplotlib.pyplot as plt
import os


def show_or_save(show, problem, plotname):
    dirname = os.path.dirname(problem)
    if not dirname:
        dirname = os.path.dirname(__file__)
    probname = os.path.basename(problem)
    if show:
        plt.show()
    else:
        if not os.path.exists(f'{dirname}/fig'):
            os.mkdir(f'{dirname}/fig')
        plt.savefig(f'{dirname}/fig/{probname}_{plotname}.png')
