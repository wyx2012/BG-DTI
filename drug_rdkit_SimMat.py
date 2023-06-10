import pandas as pd
import numpy as np
import os
# from rdkit.Chem import MACCSkeys
# from pandas import DataFrame
# from rdkit import Chem
# from rdkit.Chem.Draw import IPythonConsole
# from rdkit.Chem.Fingerprints import FingerprintMols
# from rdkit.Chem import AllChem
# from rdkit import DataStructs
# from rdkit.Chem import Draw
# from rdkit.Chem.AtomPairs import Torsions
# from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Draw
from rdkit import rdBase, Chem, DataStructs
# from rdkit.Chem.Fingerprints import FingerprintMols
# from rdkit.Chem.AtomPairs import Pairs, Torsions
# from rdkit.Chem.Draw import SimilarityMaps
dg_dg_path = 'data/drug_drug.csv'
dg_smiles_path = 'data/durg_smiles.csv'
protein_fas = pd.read_csv(dg_dg_path,header=None,index_col=None).values  #在这里填写待计算的表名
final = [[format(0, '.6f') for i in range(1,len(protein_fas))]for j in range(1,len(protein_fas))]
final = np.array(final)
prefix = "./"
# * 索引从0开始
di = pd.read_csv(os.path.join(prefix, "./data/protein_fasta.csv"), encoding='utf-8', delimiter=',', names=['dname', 'smile'])
df_smiles = di['smile'].tolist()
df_name = di['dname'].tolist()
fps = []
for x in di['smile']:
    mol = Chem.MolFromSmiles(x)
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    fps.append(fp)

# 计算数据集中第一个分子的MACC分子指纹
# 基于MACC指纹相似性的比对，获得Tanimoto系数
# the list for the dataframe
qu, ta, sim = [], [], []
# 比较所有fp成对，无重复项
for n in range(len(fps) - 1):  # -1 所以最后一个fp将不会被使用
    s = DataStructs.BulkTanimotoSimilarity(fps[n], fps[n + 1:])  # +1 与下一个到最后一个fp进行比较

    # 收集 the SMILES and values
    for m in range(len(s)):
        # qu.append(df_name[n])
        # ta.append(df_name[n + 1:][m])
        # sim.append(s[m])
        final[n][m+n+1] = str(s[m])
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
print("测试药物相似性结束！")
# d = {'query': qu, 'target': ta, 'Similarity': sim}
#
# df_final = pd.DataFrame(data=d)
# df_final = df_final.sort_values('Similarity', ascending=False)
# df_final.to_csv("./protein_smile_Test.csv", encoding="utf-8", header=False, index=False)
result_file = pd.DataFrame(final)
result_file.to_csv('data/sim/protein_sim_smail.csv',mode='a',
                   index=False,header=False,float_format='%.3f',encoding='utf-8')