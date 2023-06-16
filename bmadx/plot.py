import matplotlib.pyplot as plt

def plot_projections(coords, labels, bins, background=True):
    """Plot of coordinates projected into every possible 2D plane."""

    n_coords = coords.shape[0]

    fig, ax = plt.subplots(n_coords, n_coords, figsize=(10,10))
    mycmap = plt.get_cmap('viridis')
    mycmap.set_under(color='white') # map 0 to this color

    for i in range(n_coords):

        ax[n_coords-1,i].set_xlabel(labels[i])

        if i>0:
            ax[i,0].set_ylabel(labels[i], rotation=0)

        ax[i,i].hist(coords[i],
                     bins=bins,
                     range=(coords[i].min(), coords[i].max()))
        
        ax[i,i].yaxis.set_tick_params(left=False, labelleft=False)
        if i!= n_coords-1:
            ax[i,i].xaxis.set_tick_params(labelbottom=False)

        for j in range(i+1, n_coords):

            ax[j,i].hist2d(coords[i],
                           coords[j],
                           bins = bins,
                           range=[[coords[i].min(), coords[i].max()],
                                  [coords[j].min(), coords[j].max()]],
                           cmap = mycmap,
                           vmin = not background)
            
            ax[j,i].get_shared_x_axes().join(ax[j,i], ax[i,i])

            ax[i,j].set_visible(False)

            if i != 0:
                ax[j, i].yaxis.set_tick_params(labelleft=False)
            
            if j != n_coords-1:
                ax[j,i].xaxis.set_tick_params(labelbottom=False)

    fig.tight_layout()