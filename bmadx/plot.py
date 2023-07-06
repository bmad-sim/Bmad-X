import numpy as np
import matplotlib.pyplot as plt

SPACE_COORDS = ('x', 'y', 'z')
MOMENTUM_COORDS = ('px', 'py', 'pz')

def plot_projections(
        particles,
        coords = ('x', 'px', 'y', 'py', 'z', 'pz'),
        bins = 50,
        scale = 1e3,
        background = False,
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
        scale factor for coordinates. 1e3 for milimeters and miliradians,
        and 1 for meters and radians.
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
        all_coords.append(getattr(particles, coord)*scale)
    
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
        elif custom_lims.shape() == (n_coords, 2):
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

        ax[n_coords-1,i].set_xlabel(f'{x_coord} ({x_coord_unit})')

        if i>0:
            ax[i,0].set_ylabel(f'{x_coord} ({x_coord_unit})')

        x_array = getattr(particles, x_coord)*scale

        ax[i,i].hist(x_array,
                     bins=bins,
                     range=([coord_min[i], coord_max[i]]))
        
        ax[i,i].yaxis.set_tick_params(left=False, labelleft=False)

        if i!= n_coords-1:
            ax[i,i].xaxis.set_tick_params(labelbottom=False)

        for j in range(i+1, n_coords):

            y_coord = coords[j]
            y_array = getattr(particles, y_coord)*scale
            
            ax[j,i].hist2d(x_array,
                           y_array,
                           bins = bins,
                           range=[[coord_min[i], coord_max[i]],
                                  [coord_min[j], coord_max[j]]],
                           cmap = mycmap,
                           vmin = not background)
            
            ax[j,i].get_shared_x_axes().join(ax[j,i], ax[i,i])

            ax[i,j].set_visible(False)

            if i != 0:
                ax[j, i].yaxis.set_tick_params(labelleft=False)
            
            if j != n_coords-1:
                ax[j,i].xaxis.set_tick_params(labelbottom=False)

    fig.tight_layout()

    return fig, ax