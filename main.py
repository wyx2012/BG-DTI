import time
import datetime
import numpy as np
import torch.optim as optim
import torch
from config import Config
from utils.util import Helper
from model.CommonModel import Common_model
from model.PredictModel import Predict_model
from dataset import HetrDataset
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import evaluation_scores
kf = KFold(n_splits = 5, shuffle = True)
results = []
def train_common_model(config,helper,model,hetrdataset):
    optimizer = optim.Adam(model.parameters(),config.common_learn_rate)
    model.train()
    print("common model begin training----------",datetime.datetime.now())
    #common_loss
    for e in range(config.common_epochs):
        common_loss = 0
        begin_time = time.time()
        h_get_train_batch = hetrdataset.get_train_batch(config.batch_size);
        for i, (dg,pt,tag,dg_index,pt_index) in enumerate(h_get_train_batch):
            dg = helper.to_longtensor(dg,config.use_gpu)
            pt = helper.to_longtensor(pt,config.use_gpu)
            #common_loss
            optimizer.zero_grad()
            data,cd_pair= hetrdataset.datasetGcn()
            score,smi_common,fas_common = model(dg, pt, data)
            distance_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
            distance_loss = distance_loss(score.cuda(), data['drug_protein'].cuda())#92.204
            common_loss += distance_loss
            print(i,"distance_loss:",distance_loss)
            distance_loss.requires_grad_(True)
            distance_loss.backward()
            optimizer.step()
        #end a epech
        print("the loss of common model epoch[%d / %d]:is %4.f, time:%d s" % (e+1,config.common_epochs,common_loss,time.time()-begin_time))
        return model
if __name__=='__main__':

    ave_acc = 0
    ave_prec = 0
    ave_sens = 0
    ave_f1_score = 0
    ave_mcc = 0
    ave_auc = 0
    ave_auprc = 0
    # initial parameters class
    config = Config()

    # initial utils class
    helper = Helper()

    #initial data
    hetrdataset = HetrDataset()
    #torch.backends.cudnn.enabled = False
    model_begin_time = time.time()
    for t, (dg, pt, tag, dg_index, pt_index) in enumerate(
            hetrdataset.get_train_batch(config.batch_size)):
        dg = helper.to_longtensor(dg, config.use_gpu)
        pt = helper.to_longtensor(pt, config.use_gpu)
    #initial presentation model
    data_init,cd_pair = hetrdataset.datasetGcn()
    c_model = Common_model(config,data_init,dg,pt)
    p_model = Predict_model()
    if config.use_gpu:
        c_model = c_model.cuda()
        p_model = p_model.cuda()
    for epoch in range(config.num_epochs):
        data, cd_pairs = hetrdataset.datasetGcn()
        ListAUC = []
        ListAUPR = []
        for train_index, test_index in kf.split(cd_pairs):
            c_dmatix, train_cd_pairs, test_cd_pairs = HetrDataset.C_Dmatix(cd_pairs, train_index, test_index)
            print("         epoch:", str(epoch), "zzzzzzzzzzzzzzzz")
            model = train_common_model(config, helper, c_model, hetrdataset)
            model.eval()
            dataFS = hetrdataset.get_train_batch( config.batch_size)
            for i, (dg, pt, tag, dg_index, pt_index) in enumerate(dataFS):
                dg = helper.to_longtensor(dg, config.use_gpu)
                pt = helper.to_longtensor(pt, config.use_gpu)
                with torch.no_grad():
                    score, cir_fea, dis_fea = model(dg, pt, data)
            cir_fea = cir_fea.cpu().detach().numpy()
            dis_fea = dis_fea.cpu().detach().numpy()

            train_dataset = HetrDataset.new_dataset(cir_fea, dis_fea, train_cd_pairs)
            test_dataset = HetrDataset.new_dataset(cir_fea, dis_fea, test_cd_pairs)
            X_train, y_train = train_dataset[:, :-2], train_dataset[:, -2:]
            X_test, y_test = test_dataset[:, :-2], test_dataset[:, -2:]
            print(X_train.shape, X_test.shape)
            clf = RandomForestClassifier(n_estimators=200, n_jobs=11, max_depth=20)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred = y_pred[:, 0]
            y_prob = clf.predict_proba(X_test)
            y_prob = y_prob[1][:, 0]

            tp, fp, tn, fn, acc, prec, sens, f1_score, MCC, AUC, AUPRC = evaluation_scores.calculate_performace(
                len(y_pred), y_pred, y_prob, y_test[:, 0])

            AUCs = '%.3f' % AUC
            AUPRCs = '%.3f' % AUPRC

            ListAUC.append(AUC)

            ListAUPR.append(AUPRC)

            numAUC = np.mean(ListAUC)
            numAUPR = np.mean(ListAUPR)
            with open('results/cTest.txt', 'a') as f:
                f.write('\t  AUC = \t' + str(AUCs) + '\t  AUPRC = \t' + str(
                    AUPRCs) +'\t  time:'+str(datetime.datetime.now())+'\n')
        with open('results/cTest.txt', 'a') as f:
            f.write('\t  MeanAUC = \t' + str(numAUC) + '\t  MeanAUPRC = \t' + str(
                numAUPR) + '\n')
