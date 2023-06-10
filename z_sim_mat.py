import pandas as pd
import numpy as np
#所有待计算相似度文件名
dg_ds_path = 'data/drug_disease.csv'
dg_dg_path = 'data/drug_drug.csv'
dg_pt_path = 'data/drug_protein.csv'
dg_se_path = 'data/drug_se.csv'
pt_ds_path = 'data/protein_disease.csv'
pt_pt_path = 'data/protein_protein.csv'
smi_dict_path = 'data/smi_dict.pickle'
fas_dict_path = 'data/fas_dict.pickle'
dg_smiles_path = 'data/durg_smiles.csv'
pt_fasta_path = 'data/protein_fasta.csv'

#region 计算相似度
#读取
protein_fas = pd.read_csv(dg_ds_path,header=None,index_col=None).values  #在这里填写待计算的表名
final = [[format(0, '.6f') for i in range(1,len(protein_fas)+1)]for j in range(1,len(protein_fas)+1)]
final = np.array(final)

#计算
for ind, val in enumerate(protein_fas):
    if ind < len(protein_fas):
        for ind1, val1 in enumerate(protein_fas[ind + 1:]):
            print("第",ind+1,"行是：",val,"第",ind1+ind+2,"行是：",val1)
            #同有1的个数：
            sim_yu = np.sum((val == val1) & (val == "1"))
            #共有1的个数：
            sim_huo = np.sum((val == "1") | (val1 == "1"))
            score = format(sim_yu/sim_huo, '.6f')
            zore = format(0, '.6f')
            Jscore = zore if sim_huo == 0 else score
            final[ind+1 - 1][ind1+ind+2 - 1] = str(Jscore)
#把对角线变为1：
for i in range(final.shape[0]):
    for j in range(final.shape[1]):
        if i == j:
            final[i][j] = format(1, '.6f')
#把左下角的空缺对称填补
for i in range(final.shape[0]):
    for j in range(final.shape[1]):
        if i > j:
            final[i][j] = final[j][i]

#储存
result_file = pd.DataFrame(final)
result_file.to_csv('data/drug_di_sim.csv',mode='a',index=False,header=False,float_format='%.3f',encoding='utf-8')

#endregion








