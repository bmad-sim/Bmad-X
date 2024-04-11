import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

SPACE_COORDS = ('x', 'y', 'z')
MOMENTUM_COORDS = ('px', 'py', 'pz')
LABELS = {
    'x': 'x', 
    'px': 'p_x',
    'y': 'y', 
    'py': 'p_y', 
    'z': 'z', 
    'pz': 'p_z'
    }


def plot_projections(
        particles,
        coords = ('x', 'px', 'y', 'py', 'z', 'pz'),
        bins = 50,
        scale = 1e3,
        background = 0,
        same_lims = False,
        custom_lims = None
        ):
    """
    Plot of coordinates projected into every possible 2D plane.
    
    Parameters
    ----------
    particles: bmadx.Particle
        beam to be plotted. If you have a bmadx_torch.Beam object, 
        you can use the numpy_particle() method to get the bmadx.Particle 

    coords: array-like
        coordinates that will be plotted. Should be a 
        subset of ('x', 'px', 'y', 'py', 'z', 'pz'). 
        Default: ('x', 'px', 'y', 'py', 'z', 'pz')

    bins: int
        number of bins in histograms.
        Default: 50
    
    scale: float
        scale factor for coordinates (except pz, which is always in %).
        1e3 for milimeters and miliradians, and 1 for meters and radians. 
        Default: 1e3

    background: bool
        if False, 0 frequency pixel of 2d histograms is converted to white.
        Default: False
    
    same_lims: bool
        if True, all coords will have the same limits given by the
        largest and lowest values in all coords.
        Default: False
    
    custom_lims: array 
        if provided, sets the lims of histograms for each coords. 
        if same_lims is Frue, custom lims should have shape 2
        providing min and max for every coord.
        if same_lims is False, custom lims should have shape
        (n_coords x 2).
        Default: None

    Returns
    -------
    fig and ax pyplot objects with the projections

    """

    n_coords = len(coords)

    fig_size = (n_coords*2,) * 2

    fig, ax = plt.subplots(n_coords, n_coords, figsize=fig_size)
    mycmap = plt.get_cmap('viridis')
    mycmap.set_under(color='white') # map 0 to this color

    all_coords = []
    
    for coord in coords:
        all_coords.append(getattr(particles, coord))
    
    all_coords = np.array(all_coords)
    
    if same_lims:
        if custom_lims is None:
            coord_min = np.ones(n_coords)*all_coords.min()
            coord_max = np.ones(n_coords)*all_coords.max()
        elif len(custom_lims) == 2:
            coord_min = np.ones(n_coords)*custom_lims[0]
            coord_max = np.ones(n_coords)*custom_lims[1]
        else:
            raise ValueError("custom lims should have shape 2 when same_lims=True")
    else:
        if custom_lims is None:
            coord_min = all_coords.min(axis=1)
            coord_max = all_coords.max(axis=1)
        elif custom_lims.shape == (n_coords, 2):
            coord_min = custom_lims[:,0]
            coord_max = custom_lims[:,1]
        else:
            raise ValueError("custom lims should have shape (n_coords x 2) when same_lims=False")

    for i in range(n_coords):

        x_coord = coords[i]

        if x_coord in SPACE_COORDS and scale==1e3:
            x_coord_unit = 'mm'
        elif x_coord in SPACE_COORDS and scale==1:
            x_coord_unit = 'm'
        elif x_coord in MOMENTUM_COORDS and scale==1e3:
            x_coord_unit = 'mrad'
        elif x_coord in MOMENTUM_COORDS and scale==1:
            x_coord_unit = 'rad'
        else:
            raise ValueError("""scales should be 1 or 1e3,
            coords should be a subset of ('x', 'px', 'y', 'py', 'z', 'pz')
            """)

        if x_coord=='pz':
            x_array = getattr(particles, x_coord)*100
            ax[n_coords-1,i].set_xlabel(f'${LABELS[x_coord]}$ (%)')
            min_x = coord_min[i]*100
            max_x = coord_max[i]*100
            if i>0:
                ax[i,0].set_ylabel(f'${LABELS[x_coord]}$ (%)')

        else:
            x_array = getattr(particles, x_coord)*scale
            ax[n_coords-1,i].set_xlabel(f'${LABELS[x_coord]}$ ({x_coord_unit})')
            min_x = coord_min[i]*scale
            max_x = coord_max[i]*scale
            if i>0:
                ax[i,0].set_ylabel(f'${LABELS[x_coord]}$ ({x_coord_unit})')

        ax[i,i].hist(x_array,
                     bins=bins,
                     range=([min_x, max_x]))
        
        ax[i,i].yaxis.set_tick_params(left=False, labelleft=False)

        if i!= n_coords-1:
            ax[i,i].xaxis.set_tick_params(labelbottom=False)

        for j in range(i+1, n_coords):

            y_coord = coords[j]

            if y_coord=='pz':
                y_array = getattr(particles, y_coord)*100
                min_y = coord_min[j]*100
                max_y = coord_max[j]*100

            else:
                y_array = getattr(particles, y_coord)*scale
                min_y = coord_min[j]*scale
                max_y = coord_max[j]*scale
            
            ax[j,i].hist2d(x_array,
                           y_array,
                           bins = bins,
                           range=[[min_x, max_x],
                                  [min_y, max_y]],
                           cmap = mycmap,
                           vmin = background)
            
            ax[j,i].sharex(ax[i,i])

            ax[i,j].set_visible(False)

            if i != 0:
                ax[j, i].yaxis.set_tick_params(labelleft=False)
            
            if j != n_coords-1:
                ax[j,i].xaxis.set_tick_params(labelbottom=False)

    fig.tight_layout()
    
    return fig, ax