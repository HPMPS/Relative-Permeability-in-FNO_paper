
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
r_k = pd.read_excel("D:/Reaserch work/coarse grid paper/3d to 4d/relative K.xlsx",usecols=[0], header=None)
r_k1 = pd.read_excel("D:/Reaserch work/coarse grid paper/3d to 4d/relative K.xlsx",usecols=[1],header=None)
r_k2= pd.read_excel("D:/Reaserch work/coarse grid paper/3d to 4d/relative K.xlsx",usecols=[2],header=None)
print(r_k)
df_li = r_k.values.tolist()
x_result = []
for s_li in df_li:
    x_result.append(s_li[0])

print(x_result)
print(len(x_result))

df_li1 = r_k1.values.tolist()
x1_result = []
for s_li1 in df_li1:
    x1_result.append(s_li1[0])

print(len(x1_result))
print(x1_result)

df_li2 = r_k2.values.tolist()
x2_result = []
for s_li2 in df_li2:
    x2_result.append(s_li2[0])

print(len(x2_result))
print(x2_result)

fig = plt.figure(figsize=(5, 4))
ax1 = fig.add_subplot(111)
ax1.plot(x_result, x1_result,'r',label="krg")
ax1.legend(loc=1,prop={'family' : 'Arial', 'size' : 9})
ax1.set_ylim([0, 0.5])
ax1.set_xlabel('Gas satruation',fontproperties='Arial', size=9,fontweight='bold')
ax1.set_ylabel('Relative permeability',fontproperties='Arial', size=9,fontweight='bold')
ax2 = ax1.twinx() # this is the important function
ax2.plot(x_result, x2_result, 'b',label = "krog")
ax2.legend(loc=2,prop={'family' : 'Arial', 'size' : 9})
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 0.5])

plt.savefig("D:/Reaserch work/coarse grid paper/3d to 4d/Relative permeability.png", dpi=300)
plt.show()