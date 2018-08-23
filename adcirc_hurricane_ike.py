
# coding: utf-8

# # Read ADCIRC output from Zarr (cloud-optimized NetCDF)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from dask.distributed import Client, progress, LocalCluster
from dask_kubernetes import KubeCluster
import xarray as xr
import gcsfs
import numpy as np


# Start a dask cluster to crunch the data

# In[ ]:


import dask
import os
dask.config.config['kubernetes']['worker-template']['spec']['containers'][0]['image'] = os.environ['JUPYTER_IMAGE_SPEC']


# In[ ]:


cluster = KubeCluster()


# In[ ]:


cluster


# In[ ]:


client = Client(cluster)


# In[ ]:


fs = gcsfs.GCSFileSystem(token=None)
gcsmap = gcsfs.mapping.GCSMap('pangeo-data/rsignell/adcirc_test01', gcs=fs, check=False, create=False)


# In[ ]:


ds = xr.open_zarr(gcsmap)


# In[ ]:


ds


# In[ ]:


ds['zeta']


# In[ ]:


ds['zeta'].nbytes/1.e9


# In[ ]:


ds['zeta'].max(dim='time')


# In[ ]:


max_var = ds['zeta'].max(dim='time').persist()
progress(max_var)


# ### Visualize data on mesh using Datashader

# In[ ]:


import numpy as np
import datashader as dshade
import holoviews as hv
import geoviews as gv
import cartopy.crs as ccrs

from holoviews.operation.datashader import datashade, rasterize
from colorcet import cm_n
from matplotlib.cm import jet

datashade.precompute = True

hv.extension('bokeh')
get_ipython().run_line_magic('opts', 'Image RGB VectorField [width=800 height=600]')


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import pandas as pd


# In[ ]:


v = np.vstack((ds['x'], ds['y'], max_var)).T
verts = pd.DataFrame(v, columns=['x','y','z'])


# In[ ]:


points = gv.operation.project_points(gv.Points(verts, vdims=['z']))


# In[ ]:


tris = pd.DataFrame(ds['element'].values.astype('int')-1, columns=['v0','v1','v2'])


# In[ ]:


tiles = gv.WMTS('https://maps.wikimedia.org/osm-intl/{Z}/{X}/{Y}@2x.png')
value = 'max water level'
label = '{} (m)'.format(value)
trimesh = gv.TriMesh((tris, points), label=label)


# In[ ]:


get_ipython().run_cell_magic('opts', 'Image [colorbar=True] (cmap=jet)', "\nmeshes = rasterize(trimesh,aggregator=dshade.mean('z'))\ntiles * meshes")


# ### Extract a time series at a specified lon, lat location

# In[ ]:


# find the indices of the points in (x,y) closest to the points in (xi,yi)
def nearxy(x,y,xi,yi):
    ind = np.ones(len(xi),dtype=int)
    for i in range(len(xi)):
        dist = np.sqrt((x-xi[i])**2+(y-yi[i])**2)
        ind[i] = dist.argmin()
    return ind


# In[ ]:


#just offshore of Galveston
lat = 29.2329856
lon = -95.1535041


# In[ ]:


ind = nearxy(ds['x'].values,ds['y'].values,[lon], [lat])


# In[ ]:


get_ipython().run_cell_magic('time', '', "ds['zeta'][:,ind].plot()")

