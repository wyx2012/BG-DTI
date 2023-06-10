import pandas as pd
import numpy as np
import os
import Levenshtein
from rdkit.Chem import AllChem, Draw
from rdkit import rdBase, Chem, DataStructs

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
protein_fas = pd.read_csv(pt_pt_path,header=None,index_col=None).values  #在这里填写待计算的表名
final = [[format(0, '.6f') for i in range(1,len(protein_fas))]for j in range(1,len(protein_fas))]
final = np.array(final)
prefix = "./"
# * 索引从0开始
di = pd.read_csv(os.path.join(prefix, "./data/protein_fasta.csv"), encoding='utf-8', delimiter=',', names=['dname', 'smile'])
df_smiles = di['smile'].tolist()
df_name = di['dname'].tolist()
fps = []


for i in range(di.shape[0]):
    for j in range(di.shape[0]):
        s1=di['smile'][i]
        s2 = di['smile'][j]
        ratio = Levenshtein.ratio(s1, s2)
        final[i][j] = ratio


# #把对角线变为1：
# for i in range(final.shape[0]):
#     for j in range(final.shape[1]):
#         if i == j:
#             final[i][j] = format(1, '.6f')
# #把左下角的空缺对称填补
# for i in range(final.shape[0]):
#     for j in range(final.shape[1]):
#         if i > j:
#             final[i][j] = final[j][i]
print("测试蛋白质相似性结束！")
# d = {'query': qu, 'target': ta, 'Similarity': sim}
#
# df_final = pd.DataFrame(data=d)
# df_final = df_final.sort_values('Similarity', ascending=False)
# df_final.to_csv("./protein_smile_Test.csv", encoding="utf-8", header=False, index=False)
result_file = pd.DataFrame(final)
result_file.to_csv('data/sim/protein_sim_smail.csv',mode='a',
                   index=False,header=False,float_format='%.3f',encoding='utf-8')