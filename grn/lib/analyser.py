import matplotlib.pyplot as plt
from lib.preprocess import preprocess_progenitor_size, preprocess_whole_size
from lib.utils import shrink_and_align_stats


def show_curve_progenitor(stats, ref, max_step=None, show=True):
    """
    The evaluation function for progenitor population size.
    The lower is the better
    :param stats: the stats of the bb after running
    """
    stats, ref = shrink_and_align_stats(stats, ref, max_step=max_step)
    
    x, y = preprocess_progenitor_size(stats, ref)
    
    plt.plot(ref.index, x, label="Reference Prog")
    plt.plot(ref.index, y, label="Simulation Prog")
    
    plt.legend()
    
    if show:
        plt.show()
        
        
def show_curve(stats, ref, max_step=None, show=True):
    """
    The evaluation function for progenitor population size.
    The lower is the better
    :param stats: the stats of the bb after running
    """
    stats, ref = shrink_and_align_stats(stats, ref, max_step=max_step)
    
    x, y = preprocess_progenitor_size(stats, ref)
    
    plt.plot(ref.index, x, label="Reference Prog")
    plt.plot(ref.index, y, label="Simulation Prog")
    
    x, y = preprocess_whole_size(stats, ref)
    
    plt.plot(ref.index, x, label="Reference Whole")
    plt.plot(ref.index, y, label="Simulation Whole")
    
    plt.legend()
    
    if show:
        plt.show()