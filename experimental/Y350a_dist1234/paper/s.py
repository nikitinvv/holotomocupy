import numpy as np


# ## Parameters

# In[ ]:


n = 2048
ntheta = 4500
detector_pixelsize = 1.4760147601476e-6 * 2
energy = 17.1
wavelength = 1.24e-09 / energy
focustodetectordistance = 1.217

sx0 = -3.135e-3
z1 = np.array([5.110, 5.464, 6.879, 9.817, 10.372, 11.146, 12.594, 17.209]) * 1e-3 - sx0
z1_ids = np.array([0, 1, 2, 3]) ### note using index starting from 0
str_z1_ids = ''.join(map(str, z1_ids + 1)) 
z1 = z1[z1_ids]
ndist = len(z1)
z2 = focustodetectordistance - z1

distances = (z1 * z2) / focustodetectordistance
magnifications = focustodetectordistance / z1
norm_magnifications = magnifications / magnifications[0]