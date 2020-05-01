
import matplotlib.pyplot as plt
from matplotlib import colors

def plot_matrix(mat):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#8a6801'])
    norm = colors.Normalize(vmin=0, vmax=9)
    plt.imshow(mat, cmap=cmap, norm=norm)
    plt.show()

def plot_loss(log, metric = 'loss', save = False):
    """
    plots loss from Logger object
    """
    plt.plot(range(len(log.loss[metric])), log.loss[metric])
    plt.title('training {}'.format(metric))
    if save:
        plt.savefig('data/{}.PNG'.format(log.name))
    plt.show()


