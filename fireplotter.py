import matplotlib.pyplot as plt
from landlab.plot import graph
from landlab import imshow_grid
import numpy as np


def fireplot(grid, ngrid, K_sed, K_sed_boost, K_sed0):
    """
    Plots the topography with a pink colormap.
    Areas of elevated erodibility from fire have instead a grey colormap.
    The channel network is overlaid also using `landlab.plot.graph`.
    * grid is the landlab grid.
    * ngrid is the network model grid output by `create_network_from_raster`.
        this comes from the module createnetwork.py
    * K_sed is the sediment erodibility matrix modified by the fire model
    * K_sed_boost is the amount by which fires initially increase the sediment erodibility
    * K_sed0 is the equilibrium sediment erodibility in the absence of fires.
    """


    pinks = plt.get_cmap('pink') # pink colormap
    greys = plt.get_cmap('binary_r') # grey colormap

    # set up a blank image
    im = np.zeros(shape=(*grid.shape,4)) # rgba format. One pixel per grid cell

    # get the elevations from the grid
    z = grid.at_node['topographic__elevation']
    zs = ( z - z.min() ) / ( z.max() - z.min() ) # scale the elevations for plotting.
    zs = zs.reshape(*grid.shape)

    # determine "burned" locations in the image
    tol = K_sed_boost/np.e + K_sed0 # tolerance for "burned"
    mask = ( K_sed > tol ).flatten() # mask of places "burned"

    # burn severity...
    s = (K_sed-K_sed0)/(K_sed_boost-K_sed0)
    s[s>1]=1 # filter out overlapping burns

    s = s.reshape(*s.shape,-1)*0.5
    im = (1-s)**1.5*pinks(zs) + s**1.5*greys(zs)

    # get colors for burned locations from the grey colormap
    #bvals = greys(zs[mask])
    
    # get colors for not burned locations from the pink colormap
    #vals = pinks(zs[~mask])

    # fill in colors within the image
    #smask = np.stack((mask.reshape(*grid.shape),)*4,-1) # mask arranged to get rgba pixels.
    #im[smask] = bvals.flatten() # fill in burned colors
    #im[~smask] = vals.flatten() # fill in unburned colors

    # fill in all edges with the color black
    # https://stackoverflow.com/questions/48097068/how-to-get-boundaries-of-an-numpy-array
    edge_mask = np.ones(im[:,:,0].shape, dtype=bool)
    edge_mask[im[:,:,0].ndim * (slice(1, -1),)] = False
    im[edge_mask]=np.array([0,0,0,1]) # all edge nodes are black.
    
    # plot the image..
    x = np.arange(0,grid.spacing[0]*grid.shape[0],grid.spacing[0])
    y = np.arange(0,grid.spacing[1]*grid.shape[1],grid.spacing[1])
    plt.imshow(im, extent=[x.min(),x.max(),y.min(),y.max()],origin='lower')

    # plot the channel network
    graph.plot_links(ngrid, with_id=False, color=[0,0,1, 0.3])

    fig = plt.gcf()
    fig.set_size_inches(8,8)
    fig.patch.set_facecolor('white') # control plot properties

def fluxyplot(grid, K_sed, K_sed_boost, K_sed0, qmean, qvar):
    """
    produces a plot of the sediment flux. Here qmean and qvar are measures of average sediment flux and its variance.
    To make nice plots, we use median background subtraction to determine qmean and qvar. 
    This allows us to subtract the "static" background away from the flux map in order to show boosts to the flux from fire.
    * K_sed = the 2d numpy array of erodibilities in the watershed
    * K_sed_boost = the amount by which the erodibility is initially increased by a wildfire
    * K_sed0 = the baseline erodibility of the watershed in the absence of fire
    * qmean = a measure of the typical sediment flux as a 2d numpy array
    * qvar = a measure of the typical variability in the sediment flux as a 2d numpy array
    """

    blue = plt.get_cmap('Blues_r') # a colormap. feed me values between 0 and 1 to get various shades of blue

    # sediment flux for coloring
    q = grid.at_node['sediment__flux'].reshape(grid.shape) # the sediment flux as a 2d numpy array.
    q = (q-qmean)/np.sqrt(qvar) # rescale the flux for coloring. 
    # this subtracts out the mean and scales by the standard deviation to get a number of order 1.
    q = 1/(1+np.exp(-2*q)) # a logistic function to squash the values of q between 0 and 1 as required by "blue"

    # burn severity    
    s = (K_sed-K_sed0)/(K_sed_boost-K_sed0) # a scaled measure of burn severity between 0 and 1
    # technically if fires overlap this can actually take on values of 2 or 3... so...
    # filter out overlapping burns (which otherwise become white in the visualization)
    s[s>1]=1
    s = np.stack((s,)*4,-1)*0.5 # it needs to have the proper shape of an rgba image.


    # blend blue flux colormap where not burned with flat grey where burned.
    im = (1-s)**2*blue(q) + s**2*np.array([169/255,169/255,169/255,1])
    
    fig = plt.figure()
    x = np.arange(0,grid.spacing[0]*grid.shape[0],grid.spacing[0])
    y = np.arange(0,grid.spacing[1]*grid.shape[1],grid.spacing[1])
    plt.imshow(im, extent=[x.min(),x.max(),y.min(),y.max()],origin='lower')


    plt.title('Sediment flux in the basin with fire overlay',fontsize=12)
    plt.xlabel("y Direction [m]",fontsize=12)
    plt.ylabel("x Direction [m]",fontsize=12)
    fig.set_size_inches(8,8)
    fig.patch.set_facecolor('white') # control plot properties

