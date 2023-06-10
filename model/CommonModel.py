from torch import nn
import torch
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
import numpy as np
from config import Config
class Common_model(nn.Module):

    def __init__(self, config,data,dg,pt):
        super(Common_model, self).__init__()
        self.dg = dg
        self.dp_tensor_turn = nn.Linear(config.dr_nums+config.pt_nums, config.common_size)
        self.drug_shape_turn = nn.Linear(config.batch_size,config.dr_nums)
        self.pertion_shape_turn = nn.Linear(config.batch_size,config.pt_nums)
        self.double_dr_shape_turn = nn.Linear(config.dr_nums,data['dd']['data_matrix'].size(0))
        self.edge_dr = nn.Linear(config.dr_nums*2, config.dr_nums)
        self.edge_pt = nn.Linear(config.pt_nums*2, config.pt_nums)
        self.double_pt_shape_turn = nn.Linear(config.pt_nums,data['pp']['data_matrix'].size(0))
        self.ds_common = nn.Parameter(torch.FloatTensor(config.ds_nums, config.common_size), requires_grad=True)
        self.se_common = nn.Parameter(torch.FloatTensor(config.se_nums, config.common_size), requires_grad=True)
        self.ds_common.data.normal_(0, 0.01)
        self.se_common.data.normal_(0, 0.01)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.smi_emb = nn.Embedding(config.smi_dict_len + 1, config.embedding_size)
        self.smi_conv_region = nn.Conv2d(1, config.num_filters, (3, config.embedding_size), stride=1)
        self.fas_emb = nn.Embedding(config.fas_dict_len + 1, config.embedding_size)
        self.fas_conv_region = nn.Conv2d(1, config.num_filters, (3, config.embedding_size), stride=1)
        self.smi_mlp = MLP(config.num_filters,config.common_size)
        self.fas_mlp = MLP(config.num_filters,config.common_size)
        self.dp_mlp = MLP(config.dr_nums+config.pt_nums, config.common_size)


        #GCN和GAT
        self.drug1_GCN = GCNConv(32, 32)
        self.drug2_GCN = GCNConv(32, 32)


        self.pt1_GCN = GCNConv(32, 32)
        self.pt2_GCN = GCNConv(32, 32)

        self.dp1_GCN = GCNConv(32, 32)
        self.dp2_GCN = GCNConv(32, 32)

        self.drug1_GAT = GATConv(32, 32, heads=4, concat=False, edge_dim=1)
        self.pt1_GAT = GATConv(32, 32,heads=4,concat=False,edge_dim=1)
        self.dp1_GAT = GATConv(32, 32, heads=4, concat=False, edge_dim=1)

        self.cnn_cir = nn.Conv1d(in_channels=2,
                                 out_channels=256,
                                 kernel_size=(32, 1),
                                 stride=1,
                                 bias=True)
        self.cnn_dis = nn.Conv1d(in_channels=2,
                                 out_channels=256,
                                 kernel_size=(32, 1),
                                 stride=1,
                                 bias=True)


    def forward(self, smiles,fasta,data): #fasta:batch_size.15000/smiles:batch_size.1500
        config = Config()
        with torch.no_grad():
            smiles_1 = self.smi_emb(smiles)
            smiles_1 = torch.unsqueeze(smiles_1,1)
            smiles_1 = self.smi_conv_region(smiles_1)
            smiles_1 = self.padding1(smiles_1)
            smiles_1 = torch.relu(smiles_1)
            smiles_1 = self.conv(smiles_1)
            smiles_1 = self.padding1(smiles_1)
            smiles_1 = torch.relu(smiles_1)
            smiles_1 = self.conv(smiles_1)
            while smiles_1.size()[2] >= 2:
                smiles_1 = self._block(smiles_1)
            smiles_1 = smiles_1.squeeze()
            smile_common = self.smi_mlp(smiles_1)
            if smile_common.size(0) < config.batch_size:
                busize = config.batch_size - smile_common.size(0)
                oadta = np.zeros((busize, 32))
                smile_common = np.row_stack((smile_common.cpu(), oadta))
                smile_common = torch.tensor(smile_common).cuda().to(torch.float32)
            smile_common = torch.t(smile_common)
            smile_common = self.drug_shape_turn(smile_common)
            smile_common = torch.t(smile_common)


            #protein
            fasta_1 = self.fas_emb(fasta)                       #in:[smi_nums,smi_max_len] out:[smi_nums,smi_max_len,emb_size]
            fasta_1 = torch.unsqueeze(fasta_1,1)           #out:[smi_nums,1,smi_max_len,emb_size]
            # nn.Conv2d
            fasta_1 = self.fas_conv_region(fasta_1)        #out:[batch_size,num_filters,smi_max_len-3+1, 1]
            #Repeat 2 times

            fasta_1 = self.padding1(fasta_1)               #out:[batch_size,num_filters,smi_max_len, 1]
            fasta_1 = torch.relu(fasta_1)
            fasta_1 = self.conv(fasta_1)                   #out:[batch_size,num_filters,smi_max_len-3+1, 1]
            fasta_1 = self.padding1(fasta_1)               #out:[batch_size,num_filters,smi_max_len, 1]
            fasta_1 = torch.relu(fasta_1)
            fasta_1 = self.conv(fasta_1)                   #out:[batch_size,num_filters,smi_max_len-3+1, 1]
            while fasta_1.size()[2] >= 2:
                fasta_1 = self._block(fasta_1)   #maxpool/conv/conv
            fasta_1 = fasta_1.squeeze()                    #[batch_size, num_filters]
            fasta_common = self.fas_mlp(fasta_1)
            if fasta_common.size(0) < config.batch_size:
                busize1 = config.batch_size - fasta_common.size(0)
                oadta1 = np.zeros((busize1, 32))
                fasta_common = np.row_stack((fasta_common.cpu(), oadta1))
                fasta_common = torch.tensor(fasta_common).cuda().to(torch.float32)
            fasta_common = torch.t(fasta_common)
            fasta_common = self.pertion_shape_turn(fasta_common)
            fasta_common = torch.t(fasta_common)
            if fasta_common.size(0) < config.pt_nums:
                busize1 = config.pt_nums - fasta_common.size(0)
                oadta1 = np.zeros((busize1, 32))
                fasta_common = np.row_stack((fasta_common.cpu(), oadta1))
                fasta_common = torch.tensor(fasta_common).cuda().to(torch.float32)
            #消融实验
            # smile_common = torch.randn(config.dr_nums, config.common_size)
            # fasta_common = torch.randn(config.pt_nums, config.common_size)
            #drug
            x_dr_f1 = torch.relu(self.drug1_GCN(smile_common.cuda(),
                                                  data['dd']['edges'].cuda(),
                                                  data['dd']['data_matrix'][data['dd']['edges'][0],
                                                                            data['dd']['edges'][1]].cuda()))
            x_dr_att = torch.relu(self.drug1_GAT(x_dr_f1,
                                                data['dd']['edges'].cuda(),
                                                data['dd']['data_matrix'][data['dd']['edges'][0],
                                                data['dd']['edges'][1]].cuda()))
            x_dr_f2 = torch.relu(self.drug2_GCN(x_dr_att,
                                                data['dd']['edges'].cuda(),
                                                data['dd']['data_matrix'][data['dd']['edges'][0],
                                                data['dd']['edges'][1]].cuda()))
            x_dr_f1_e = torch.relu(self.drug1_GCN(smile_common.cuda(),
                                                  data['dd1']['edges'].cuda(),
                                                  data['dd1']['data_matrix'][data['dd1']['edges'][0],
                                                                            data['dd1']['edges'][1]].cuda()))
            x_dr_att_e = torch.relu(self.drug1_GAT(x_dr_f1_e,
                                                data['dd1']['edges'].cuda(),
                                                data['dd1']['data_matrix'][data['dd1']['edges'][0],
                                                data['dd1']['edges'][1]].cuda()))
            x_dr_f2_e = torch.relu(self.drug2_GCN(x_dr_att_e,
                                                data['dd1']['edges'].cuda(),
                                                data['dd1']['data_matrix'][data['dd1']['edges'][0],
                                                data['dd1']['edges'][1]].cuda()))



            #protein
            x_pt_f1 = torch.relu(self.pt1_GCN(fasta_common.cuda(),
                                                data['pp']['edges'].cuda(),
                                                data['pp']['data_matrix'][data['pp']['edges'][0],
                                                                            data['pp']['edges'][1]].cuda()))

            x_pt_att = torch.relu(self.pt1_GAT(x_pt_f1,
                                                 data['pp']['edges'].cuda(),
                                                 data['pp']['data_matrix'][data['pp']['edges'][0],
                                                 data['pp']['edges'][1]].cuda()))
            x_pt_f2 = torch.relu(self.pt2_GCN(x_pt_att,
                                                data['pp']['edges'].cuda(),
                                                data['pp']['data_matrix'][data['pp']['edges'][0],
                                                data['pp']['edges'][1]].cuda()))
            x_pt_f1_e = torch.relu(self.pt1_GCN(fasta_common.cuda(),
                                                data['pp1']['edges'].cuda(),
                                                data['pp1']['data_matrix'][data['pp1']['edges'][0],
                                                                          data['pp1']['edges'][1]].cuda()))

            x_pt_att_e = torch.relu(self.pt1_GAT(x_pt_f1_e,
                                                 data['pp1']['edges'].cuda(),
                                                 data['pp1']['data_matrix'][data['pp1']['edges'][0],
                                                                           data['pp1']['edges'][1]].cuda()))
            x_pt_f2_e = torch.relu(self.pt2_GCN(x_pt_att_e,
                                                data['pp1']['edges'].cuda(),
                                                data['pp1']['data_matrix'][data['pp1']['edges'][0],
                                                                          data['pp1']['edges'][1]].cuda()))
            x_pt_f1 = torch.add(x_pt_f1, x_pt_f1_e)
            x_pt_f2 = torch.add(x_pt_f2, x_pt_f2_e)
            #drug_portion
            dr1 = torch.cat((torch.zeros(config.dr_nums,config.dr_nums,dtype = torch.float32),data['dp']['data_matrix']), dim=1)
            dr2 = torch.cat((data['dp']['data_matrix'].t(),torch.zeros(config.pt_nums,config.pt_nums,dtype = torch.float32)), dim=1)
            dp_common1 = torch.cat((dr1, dr2), dim=0)
            dp_common1 = dp_common1.cuda()
            dp_common2 = self.dp_mlp(dp_common1)

            # 消融实验
            #dp_common2 = torch.randn(config.pt_nums+config.dr_nums, config.common_size)
            x_dp_f1 = torch.relu(self.dp1_GCN(dp_common2.cuda(),
                                                data['dp']['edges'].cuda(),#51.51
                                                data['dp']['data_matrix'][data['dp']['edges'][0],#2601一维
                                                                            data['dp']['edges'][1]].cuda()))

            x_dp_att = torch.relu(self.dp1_GAT(x_dp_f1,
                                                   data['dp']['edges'].cuda(),
                                                   data['dp']['data_matrix'][data['dp']['edges'][0],
                                                                              data['dp']['edges'][1]].cuda()))
            x_dp_f2 = torch.relu(self.dp2_GCN(x_dp_att,
                                                  data['dp']['edges'].cuda(),
                                                  data['dp']['data_matrix'][data['dp']['edges'][0],
                                                                             data['dp']['edges'][1]].cuda()))
            x_d1 = x_dp_f1[:config.dr_nums,:]
            x_p1 = x_dp_f1[config.dr_nums:,:]
            x_d2 = x_dp_f2[:config.dr_nums, :]
            x_p2 = x_dp_f2[config.dr_nums:, :]
            x_dr_f1 = torch.add(x_dr_f1, x_dr_f1_e)
            x_dr_f1 =  torch.add(x_dr_f1, x_d1)

            x_dr_f2 = torch.add(x_dr_f2, x_dr_f2_e)
            x_dr_f2 = torch.add(x_dr_f2, x_d2)

            x_pt_f1 = torch.add(x_pt_f1, x_pt_f1_e)
            x_pt_f1 = torch.add(x_pt_f1, x_p1)

            x_pt_f2 = torch.add(x_pt_f2, x_pt_f2_e)
            x_pt_f2 = torch.add(x_pt_f2, x_p2)

            X_dr = torch.cat((x_dr_f1,x_dr_f2), 1).t()  # 128.92
            X_dr = X_dr.view(1, 2, 32, -1)

            X_pt = torch.cat((x_pt_f1, x_pt_f2), 1).t()  # 64.32+64.32=64.64
            X_pt = X_pt.view(1, 2, 32, -1)  # 1.2.32.64

            X_dp = torch.cat((x_dp_f1,x_dp_f2), 1).t()
            X_dp = X_dp.view(1, 2, 32, -1)

            dr_fea = self.cnn_cir(X_dr)
            dr_fea = dr_fea.view(256, config.dr_nums)
            dr_fea = torch.t(dr_fea)
            pt_fea = self.cnn_dis(X_pt)
            pt_fea = pt_fea.view(256, config.pt_nums)
            pt_fea = torch.t(pt_fea)
            dp_featest = dr_fea.mm(pt_fea.t())
            dp_fea = self.cnn_cir(X_dp)
            dp_fea = dp_fea.view(256, config.dr_nums+config.pt_nums)
            dp_fea = torch.t(dp_fea)

        # .mm矩阵相乘
        return dr_fea.mm(pt_fea.t()),x_dr_f2, x_pt_f2


    def _block(self, x):
                                     #in: [batch_size,num_filters,smi_max_len-3+1, 1]
        x = self.padding2(x)         #out:[batch_size,num_filters,smi_max_len-1, 1]
        px = self.max_pool(x)        #out:[batch_size,num_filters,(smi_max_len-1)/2, 1]

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x

# multi-layer perceptron 多层感知器
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, output_size)
        )



    def forward(self, x):
        out = self.linear(x)
        return out