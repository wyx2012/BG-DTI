import pandas as pd
import numpy as np

th = 0.5
#region 添加边缘---药物
a = pd.read_csv('data/sim/drug_sim_smail.csv',header=None,index_col=None).values  #在这里填写待计算的表名
b = pd.read_csv('data/sim/drug_di_sim.csv',header=None,index_col=None).values  #在这里填写待计算的表名
c = pd.read_csv('data/sim/drug_drug_sim.csv',header=None,index_col=None).values  #在这里填写待计算的表名
d = pd.read_csv('data/sim/drug_se_sim.csv',header=None,index_col=None).values  #在这里填写待计算的表名
drug_drug = pd.read_csv('data/sim/drug_drug.csv',header=None,index_col=None).values  #在这里填写待计算的表名
a = a.astype(float)
b = b.astype(float)
c = c.astype(float)
drug_drug = drug_drug.astype(float)
d = d.astype(float)
a[a >= th] = 1
a[a < th] = 0

b[b >= th] = 1
b[b < th] = 0

c[c >= th] = 1
c[c < th] = 0

d[d >= th] = 1
d[d < th] = 0

Final = a+b+c+d
Final[Final >= 1] = 1
Finalj = Final-drug_drug
Finalj[Finalj==-1]=0
for i in range(Finalj.shape[0]):
    for j in range(Finalj.shape[1]):
        if i == j:
            Finalj[i][j] = format(0, '.6f')
result_file = pd.DataFrame(Finalj)
result_file.to_csv('data/add_edge/drug_add_edge.csv',mode='a',index=False,header=False,float_format='%.3f',encoding='utf-8')
#endregion


#region 添加边缘---靶标
e = pd.read_csv('data/sim/pt_ds_sim.csv',header=None,index_col=None).values  #在这里填写待计算的表名
f = pd.read_csv('data/sim/pt_pt_sim.csv',header=None,index_col=None).values  #在这里填写待计算的表名
g = pd.read_csv('data/sim/protein_sim_smail.csv',header=None,index_col=None).values  #在这里填写待计算的表名
pt_pt = pd.read_csv('data/sim/protein_protein.csv',header=None,index_col=None).values  #在这里填写待计算的表名
e = e.astype(float)
f = f.astype(float)
g = g.astype(float)

e[e >= th] = 1
e[e < th] = 0

f[f >= th] = 1
f[f < th] = 0

g[g >= th] = 1
g[g < th] = 0

Final1 = e+f+g
Final1[Final1 >= 1] = 1
Finalj1 = Final1-pt_pt
Finalj1[Finalj1==-1]=0
for i in range(Finalj1.shape[0]):
    for j in range(Finalj1.shape[1]):
        if i == j:
            Finalj1[i][j] = format(0, '.6f')

result_file1 = pd.DataFrame(Finalj1)
result_file1.to_csv('data/add_edge/protein_add_edge.csv',mode='a',index=False,header=False,float_format='%.3f',encoding='utf-8')