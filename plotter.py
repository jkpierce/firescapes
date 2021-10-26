import matplotlib.pyplot as plt
from landlab.plot import graph
import numpy as np


def fireplot(grid, ngrid, K_sed, K_sed_boost, K_sed0, zmax, zmin):
    """
    * grid is the landlab grid.
    * ngrid is the network model grid output by `create_network_from_raster`.
        this comes from the module createnetwork.py
    * K_sed is the sediment erodibility matrix modified by the fire model
    * K_sed_boost is the amount by which fires initially increase the sediment erodibility
    * K_sed0 is the equilibrium sediment erodibility in the absence of fires.
    """
    
    pinks = plt.get_cmap('pink') # pink colormap
    greys = plt.get_cmap('binary_r')# grey colormap

    # set up a blank image... 
    im = np.zeros(shape=(*grid.shape,4)) # rgba format. One pixel per grid cell

    # get the elevations from the grid
    z = grid.at_node['topographic__elevation']
    zs = (z-zmin)/(zmax-zmin) # scale the elevations for plotting.

    # determine "burned" locations in the image
    tol = K_sed_boost*0.25 + K_sed0 # tolerance for "burned"
    mask = ( K_sed > tol ).flatten() # mask of places "burned"

    # get colors for burned locations from the grey colormap
    bvals = greys(zs[mask])
    
    # get colors for not burned locations from the pink colormap
    vals = pinks(zs[~mask])

    # fill in colors within the image
    smask = np.stack((mask.reshape(*grid.shape),)*4,-1) # mask arranged to get rgba pixels.
    im[smask] = bvals.flatten() # fill in burned colors
    im[~smask] = vals.flatten() # fill in unburned colors

    # fill in all edges with the color black

        # https://stackoverflow.com/questions/48097068/how-to-get-boundaries-of-an-numpy-array
    edge = np.ones(im[:,:,0].shape, dtype=bool)
    edge[im[:,:,0].ndim * (slice(1, -1),)] = False
    im[edge]=np.array([0,0,0,1]) # all edge nodes are black.
    
    # plot the image..
    x = np.arange(0,grid.spacing[0]*grid.shape[0],grid.spacing[0])
    y = np.arange(0,grid.spacing[1]*grid.shape[1],grid.spacing[1])
    plt.imshow(im, extent=[x.min(),x.max(),y.min(),y.max()],origin='lower')
    
    # plot the channel network
    graph.plot_links(ngrid, with_id=False)

    fig = plt.gcf()
    fig.set_size_inches(8,8)
    fig.patch.set_facecolor('white') # control plot properties
