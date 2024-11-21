import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot(df, x1, x2, x_bins, y_bins):
    '''
    df = (pandas dataframe) with columns x1 and x2
    x1 = (str) name of the column x1
    x2 = (str) name of the column x2
    x_bins = (numpy array) binning for x1
    y_bins = (numpy array) binning for x2
    '''
    fig, ax_main = plt.subplots(figsize=(8, 8))
    divider = make_axes_locatable(ax_main)
    ax_top = divider.append_axes("top", 1.2, pad=0.2, sharex=ax_main)
    ax_right = divider.append_axes("right", 1.2, pad=0.2, sharey=ax_main)

    # Plot the 2D histogram in the main axes
    hist, x_edges, y_edges = np.histogram2d(x=df[x1], y=df[x2], bins=[x_bins, y_bins])
    ax_main.imshow(hist.T, origin='lower', extent=(x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()), aspect='auto', cmap='coolwarm')
    ax_main.set_xlabel(x1)
    ax_main.set_ylabel(x2)

    # Plot the marginalized histogram on top
    ax_top.hist(df[x1], bins=x_bins, color='lightblue', edgecolor='black')
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.set_yticks([])
    ax_top.xaxis.tick_top()

    # Plot the marginalized histogram on the right
    ax_right.hist(df[x2], bins=y_bins, color='lightblue', edgecolor='black', orientation='horizontal')#lightcoral
    ax_right.set_ylim(ax_main.get_ylim())
    ax_right.set_xticks([])
    ax_right.yaxis.tick_right()
    return fig
