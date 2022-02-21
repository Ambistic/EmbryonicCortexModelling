import matplotlib.pyplot as plt


def plot_2d_population(cell_list, color='blue'):
    """Plot a two dimensional population provided as a list of Cell objects."""

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for cell in cell_list:
        ax.add_patch(plt.Circle(cell.position, 0.5, color=color, alpha=0.4))
        plt.plot(cell.position[0], cell.position[1], '.', color=color)
    ax.set_aspect('equal')
    
    
def plot_2d_positions(positions, color='blue'):
    """Plot a two dimensional population provided as a list of Cell objects."""

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for position in positions:
        ax.add_patch(plt.Circle(position, 0.5, color=color, alpha=0.4))
        plt.plot(position[0], position[1], '.', color=color)
    ax.set_aspect('equal')
    
    
def plot_2d_positions_colors(positions, colors):
    """Plot a two dimensional population provided as a list of Cell objects."""

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for position, color in zip(positions, colors):
        ax.add_patch(plt.Circle(position, 0.5, color=color, alpha=0.4))
        plt.plot(position[0], position[1], '.', color=color)
    ax.set_aspect('equal')