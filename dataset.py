import pandas as pd
import numpy as np
import pickle
import torch
import csv
import random
from sklearn.model_selection import StratifiedKFold

from config import Config

class HetrDataset():
    def __init__(self):
        config = Config()
        self.repeat_nums = config.repeat_nums
        self.fold_nums = config.fold_nums
        self.neg_samp_ratio = config.neg_samp_ratio

        self.dg_smi_path = config.dg_smiles_path
        self.pt_fas_path = config.pt_fasta_path
        self.smi_dict_path = config.smi_dict_path
        self.fas_dict_path = config.fas_dict_path
        self.smi_ngram = config.smi_n_gram
        self.fas_ngram = config.fas_n_gram
        self.smi_max_len = config.smiles_max_len
        self.fas_max_len = config.fasta_max_len

        self.dg_pt_path = config.dg_pt_path

        self.dg_dg_path = config.dg_dg_path
        self.dg_ds_path = config.dg_ds_path
        self.dg_se_path = config.dg_se_path
        self.pt_ds_path = config.pt_ds_path
        self.pt_pt_path = config.pt_pt_path

        self.read_data()
        self.pre_process()

    def read_data(self):
        #sequence data
        self.drug_smi = pd.read_csv(self.dg_smi_path,header=None,index_col=None).values
        self.protein_fas = pd.read_csv(self.pt_fas_path,header=None,index_col=None).values
        #Load mapping dictionary
        with open(self.smi_dict_path, "rb") as f:
            self.smi_dict = pickle.load(f)
        with open(self.fas_dict_path, "rb") as f:
            self.fas_dict = pickle.load(f)

        self.dg_pt = pd.read_csv(self.dg_pt_path, header=0, index_col=0).values

        #Load heterogeneous information
        self.dg_dg = pd.read_csv(self.dg_dg_path,header=0,index_col=0).values
        self.dg_ds = pd.read_csv(self.dg_ds_path,header=0,index_col=0).values
        self.dg_se = pd.read_csv(self.dg_se_path,header=0,index_col=0).values
        self.pt_ds = pd.read_csv(self.pt_ds_path,header=0,index_col=0).values
        self.pt_pt = pd.read_csv(self.pt_pt_path,header=0,index_col=0).values

    def pre_process(self):
        """
        :return:all_data_set:list    repeat_nums*fold_nums*3
        """
        self.all_data_set = []
        whole_positive_index = []
        whole_negetive_index = []
        for i in range(self.dg_pt.shape[0]):
            for j in range(self.dg_pt.shape[1]):
                if int(self.dg_pt[i, j]) == 1:
                    whole_positive_index.append([i, j])
                elif int(self.dg_pt[i, j]) == 0:
                    whole_negetive_index.append([i, j])

        for x in range(self.repeat_nums):

            #Downsample negative samples
            negative_sample_index = np.random.choice(np.arange(len(whole_negetive_index)),
                                                     size=self.neg_samp_ratio * len(whole_positive_index),replace=False)
            data_set = np.zeros((self.neg_samp_ratio*len(whole_positive_index) + len(negative_sample_index),3), dtype=int)

            count = 0
            for item in whole_positive_index:
                #Oversample positive samples
                for i in range(self.neg_samp_ratio):
                    data_set[count][0] = item[0]
                    data_set[count][1] = item[1]
                    data_set[count][2] = 1
                    count = count + 1
            for i in negative_sample_index:
                data_set[count][0] = whole_negetive_index[i][0]
                data_set[count][1] = whole_negetive_index[i][1]
                data_set[count][2] = 0
                count = count + 1

            all_fold_dataset = []
            rs = np.random.randint(0,1000,1)[0]
            kf = StratifiedKFold(n_splits=self.fold_nums, shuffle=True, random_state=rs)
            for train_index, test_index in kf.split(data_set[:,0:2], data_set[:, 2]):
                train_data, test_data = data_set[train_index], data_set[test_index]
                one_fold_dataset = []
                one_fold_dataset.append(train_data)
                one_fold_dataset.append(test_data)
                all_fold_dataset.append(one_fold_dataset)

            self.all_data_set.append(all_fold_dataset)

        #Express the sequence numerically
        self.smi_input = np.zeros((len(self.drug_smi),self.smi_max_len),dtype=int)
        self.fas_input = np.zeros((len(self.protein_fas),self.fas_max_len),dtype=int)

        for i in range(len(self.drug_smi)):
            for j in range(len(self.drug_smi[i,1]) - self.smi_ngram +1):
                key = self.drug_smi[i,1][j:j + self.smi_ngram]
                self.smi_input[i,j] = self.smi_dict[key]

        for i in range(len(self.protein_fas)):
            for j in range(len(self.protein_fas[i,1]) - self.fas_ngram +1):
                key = self.protein_fas[i,1][j:j + self.fas_ngram]
                self.fas_input[i,j] = self.fas_dict[key]

    def get_train_batch(self,batch_size):

        train_drugs = []
        train_proteins = []
        train_affinity = []
        drug_index = []
        protein_index = []
        train_data = self.all_data_set[0][0][0]

        for index,(i,j,tag) in enumerate(train_data):
            train_drugs.append(self.smi_input[i])
            train_proteins.append(self.fas_input[j])
            train_affinity.append(tag)
            drug_index.append(i)
            protein_index.append(j)

        train_drugs = np.array(train_drugs)
        train_proteins = np.array(train_proteins)
        train_affinity = np.array(train_affinity)
        drug_index = np.array(drug_index)
        protein_index = np.array(protein_index)

        #Shuffle training data and labels
        data_index = np.arange(len(train_drugs))
        np.random.shuffle(data_index)
        train_drugs = train_drugs[data_index]
        train_proteins = train_proteins[data_index]
        train_affinity = train_affinity[data_index]
        drug_index = drug_index[data_index]
        protein_index = protein_index[data_index]

        #Iterative return
        sindex = 0
        eindex = batch_size
        while eindex < len(train_drugs):
            tra_dg_batch = train_drugs[sindex:eindex,:]
            tra_pt_batch = train_proteins[sindex:eindex,:]
            tra_tag_batch = train_affinity[sindex:eindex]
            dg_index_batch = drug_index[sindex:eindex]
            pt_index_batch = protein_index[sindex:eindex]

            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            yield tra_dg_batch,tra_pt_batch,tra_tag_batch,dg_index_batch,pt_index_batch

        if eindex >= len(train_drugs):
            tra_dg_batch = train_drugs[sindex:,:]
            tra_pt_batch = train_proteins[sindex:,:]
            tra_tag_batch = train_affinity[sindex:]
            dg_index_batch = drug_index[sindex:]
            pt_index_batch = protein_index[sindex:]
            yield tra_dg_batch,tra_pt_batch,tra_tag_batch,dg_index_batch,pt_index_batch

    def get_test_batch(self,repeat_nums,flod_nums,batch_size):

        train_drugs = []
        train_proteins = []
        train_affinity = []
        drug_index = []
        protein_index = []
        train_data = self.all_data_set[repeat_nums][0][1]

        for index,(i,j,tag) in enumerate(train_data):
            train_drugs.append(self.smi_input[i])
            train_proteins.append(self.fas_input[j])
            train_affinity.append(tag)
            drug_index.append(i)
            protein_index.append(j)

        train_drugs = np.array(train_drugs)
        train_proteins = np.array(train_proteins)
        train_affinity = np.array(train_affinity)
        drug_index = np.array(drug_index)
        protein_index = np.array(protein_index)

        data_index = np.arange(len(train_drugs))
        np.random.shuffle(data_index)
        train_drugs = train_drugs[data_index]
        train_proteins = train_proteins[data_index]
        train_affinity = train_affinity[data_index]
        drug_index = drug_index[data_index]
        protein_index = protein_index[data_index]

        sindex = 0
        eindex = batch_size
        while eindex < len(train_drugs):
            tra_dg_batch = train_drugs[sindex:eindex,:]
            tra_pt_batch = train_proteins[sindex:eindex,:]
            tra_tag_batch = train_affinity[sindex:eindex]
            dg_index_batch = drug_index[sindex:eindex]
            pt_index_batch = protein_index[sindex:eindex]

            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            yield tra_dg_batch,tra_pt_batch,tra_tag_batch,dg_index_batch,pt_index_batch

        if eindex >= len(train_drugs):
            tra_dg_batch = train_drugs[sindex:,:]
            tra_pt_batch = train_proteins[sindex:,:]
            tra_tag_batch = train_affinity[sindex:]
            dg_index_batch = drug_index[sindex:]
            pt_index_batch = protein_index[sindex:]
            yield tra_dg_batch,tra_pt_batch,tra_tag_batch,dg_index_batch,pt_index_batch

    def read_csv(self,path):
        with open(path, 'r', newline='') as csv_file:
            reader = csv.reader(csv_file)
            cd_data = []
            cd_data += [[float(i) for i in row] for row in reader]
            return torch.Tensor(cd_data)

    def get_edge_index(self,matrix):
        edge_index = [[], []]
        for i in range(matrix.size(0)):
            for j in range(matrix.size(1)):
                # if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
        return torch.LongTensor(edge_index)

    def datasetGcn(self):
        dataset = dict()
        dataset['drug_protein'] = self.read_csv('./data/drug_protein.csv')
        zero_index = []
        one_index = []
        cd_pairs = []
        for i in range(dataset['drug_protein'].size(0)):
            for j in range(dataset['drug_protein'].size(1)):
                if dataset['drug_protein'][i][j] < 1:
                    zero_index.append([i, j, 0])
                if dataset['drug_protein'][i][j] >= 1:
                    one_index.append([i, j, 1])
        cd_pairs = random.sample(zero_index, len(one_index)) + one_index
        dd_matrix = self.read_csv('./data/drug_drug.csv')
        dp_edge_index = self.get_edge_index(dataset['drug_protein'])
        dataset['dp'] = {'data_matrix': dataset['drug_protein'], 'edges': dp_edge_index}
        dd_edge_index = self.get_edge_index(dd_matrix)
        dataset['dd'] = {'data_matrix': dd_matrix, 'edges': dd_edge_index}

        pp_matrix = self.read_csv('./data/protein_protein.csv')
        pp_edge_index = self.get_edge_index(pp_matrix)
        dataset['pp'] = {'data_matrix': pp_matrix, 'edges': pp_edge_index}

        dd1_matrix = self.read_csv('./data/add_edge/drug_add_edge.csv')
        dd1_edge_index = self.get_edge_index(dd1_matrix)
        dataset['dd1'] = {'data_matrix': dd1_matrix, 'edges': dd1_edge_index}

        pp1_matrix = self.read_csv('./data/add_edge/protein_add_edge.csv')
        pp1_edge_index = self.get_edge_index(pp1_matrix)
        dataset['pp1'] = {'data_matrix': pp1_matrix, 'edges': pp1_edge_index}

        return dataset,cd_pairs

    def C_Dmatix(cd_pairs, trainindex, testindex):
        c_dmatix = np.zeros((708, 1512))
        for i in trainindex:
            if cd_pairs[i][2] == 1:
                c_dmatix[cd_pairs[i][0]][cd_pairs[i][1]] = 1

        dataset = dict()
        cd_data = []
        cd_data += [[float(i) for i in row] for row in c_dmatix]
        cd_data = torch.Tensor(cd_data)
        dataset['c_d'] = cd_data

        train_cd_pairs = []
        test_cd_pairs = []
        for m in trainindex:
            train_cd_pairs.append(cd_pairs[m])

        for n in testindex:
            test_cd_pairs.append(cd_pairs[n])

        return dataset['c_d'], train_cd_pairs, test_cd_pairs

    def new_dataset(cir_fea, dis_fea, cd_pairs):
        unknown_pairs = []
        known_pairs = []

        for pair in cd_pairs:
            if pair[2] == 1:
                known_pairs.append(pair[:2])

            if pair[2] == 0:
                unknown_pairs.append(pair[:2])

        print("--------------------")
        print(cir_fea.shape, dis_fea.shape)
        print("--------------------")
        print(len(unknown_pairs), len(known_pairs))

        nega_list = []
        for i in range(len(unknown_pairs)):
            nega = cir_fea[unknown_pairs[i][0], :].tolist() + dis_fea[unknown_pairs[i][1], :].tolist() + [0, 1]
            nega_list.append(nega)

        posi_list = []
        for j in range(len(known_pairs)):
            posi = cir_fea[known_pairs[j][0], :].tolist() + dis_fea[known_pairs[j][1], :].tolist() + [1, 0]
            posi_list.append(posi)

        samples = posi_list + nega_list

        random.shuffle(samples)
        samples = np.array(samples)
        return samples
