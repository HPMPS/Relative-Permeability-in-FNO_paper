import torch
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import numpy as np
import pandas as pd
pv.global_theme.font.label_size = 10
pv.global_theme.font.color = 'black'
pv.global_theme.title = 'improving FNO-net from 3D(x,y,z) to 4D(x,y,z,t)'

# load data
################################################################
from pyvista import UniformGrid
from traits.trait_types import true
import torch
import numpy as np
import pandas as pd

def getdata_k(path_kx1,path_kx2,path_kx3,path_kx4,path_kx5,path_ky1,path_ky2,path_ky3,path_ky4,path_ky5):

    data_kx1 = pd.read_excel(path_kx1,header=None)
    data_kx2 = pd.read_excel(path_kx2, header=None)
    data_kx3 = pd.read_excel(path_kx3, header=None)
    data_kx4 = pd.read_excel(path_kx4, header=None)
    data_kx5 = pd.read_excel(path_kx5, header=None)

    data_ky1 = pd.read_excel(path_ky1,header=None)
    data_ky2 = pd.read_excel(path_ky2, header=None)
    data_ky3 = pd.read_excel(path_ky3, header=None)
    data_ky4 = pd.read_excel(path_ky4, header=None)
    data_ky5 = pd.read_excel(path_ky5, header=None)

    kx1_128_128 = np.array(data_kx1)
    kx2_128_128 = np.array(data_kx2)
    kx3_128_128 = np.array(data_kx3)
    kx4_128_128 = np.array(data_kx4)
    kx5_128_128 = np.array(data_kx5)

    ky1_128_128 = np.array(data_ky1)
    ky2_128_128 = np.array(data_ky2)
    ky3_128_128 = np.array(data_ky3)
    ky4_128_128 = np.array(data_ky4)
    ky5_128_128 = np.array(data_ky5)


    ###组合模型预测的50timestep的5层
    saturation_50timestep_layer1 = kx1_128_128.reshape(128, 128)
    saturation_50timestep_layer2 = kx2_128_128.reshape(128, 128)
    saturation_50timestep_layer3 = kx3_128_128.reshape(128, 128)
    saturation_50timestep_layer4 = kx4_128_128.reshape(128, 128)
    saturation_50timestep_layer5 = kx5_128_128.reshape(128, 128)
    coupling_50timestep = np.array(
        [saturation_50timestep_layer1, saturation_50timestep_layer2, saturation_50timestep_layer3,
         saturation_50timestep_layer4, saturation_50timestep_layer5])
    coupling_50timestep_tenor = torch.from_numpy(coupling_50timestep)
    coupling_50timestep_tenor = coupling_50timestep_tenor.permute(1, 2, 0)
    coupling_50timestep_array = coupling_50timestep_tenor.numpy()
    coupling_50timestep_array = np.flip(coupling_50timestep_array, axis=-1)

    grid1 = pv.UniformGrid()
    grid1.dimensions = np.array(coupling_50timestep_array.shape) + 1
    # Edit the spatial reference
    grid1.origin = (100, 33, 55.6)  # The bottom left corner of the data set
    grid1.spacing = (1, 1, 5)  # These are the cell sizes along each axis

    grid1.cell_data["Predicted saturation field at T=50timestep"] = coupling_50timestep_array.flatten(
        order="F")  # Flatten the array!
    sargs = dict(height=1, vertical=True, position_x=0.9, position_y=0.05)
    p = pv.Plotter(border_color='white')
    p.add_mesh(grid1, cmap="rainbow",scalar_bar_args=sargs)
    p.set_background('white')

    p.show(screenshot='D:\Reaserch work\coarse grid paper/3d to 4d/128_128_kx')


k = getdata_k('D:\Reaserch work\paper in PHD\paper work 2022\permeability x/kx_layer1.xlsx','D:\Reaserch work\paper in PHD\paper work 2022\permeability x/kx_layer2.xlsx',
              'D:\Reaserch work\paper in PHD\paper work 2022\permeability x/kx_layer3.xlsx','D:\Reaserch work\paper in PHD\paper work 2022\permeability x/kx_layer4.xlsx',
              'D:\Reaserch work\paper in PHD\paper work 2022\permeability x/kx_layer5.xlsx',
             'D:\Reaserch work\paper in PHD\paper work 2022\permeability y/ky_layer1.xlsx','D:\Reaserch work\paper in PHD\paper work 2022\permeability y/ky_layer2.xlsx',
              'D:\Reaserch work\paper in PHD\paper work 2022\permeability y/ky_layer3.xlsx','D:\Reaserch work\paper in PHD\paper work 2022\permeability y/ky_layer4.xlsx',
              'D:\Reaserch work\paper in PHD\paper work 2022\permeability y/ky_layer5.xlsx')







