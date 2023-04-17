import os
import torch
import random
import pickle
import argparse
import numpy as np
import torch.nn as nn
import sys
from time import time
import logging
import random
import copy
import pywt
import math
# from network import ConvNCF
from collections import defaultdict
from scipy import stats
from math import sqrt
import torch.utils.data
from copy import deepcopy
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import average_precision_score,accuracy_score,f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity



def parse_args():
    parser = argparse.ArgumentParser(description="Run drug_side.")

    # parser = argparse.ArgumentParser(description = 'Model')
    parser.add_argument('--data_dir', nargs='?', default='./data/',
                        help='Input data path.')
    parser.add_argument('--save_dir', nargs='?', default='./1test-model/',
                        help='save_model.')
    parser.add_argument('--seed', type=int, default=2022,
                        help='Random seed.')
    parser.add_argument('--log_dir', nargs='?', default='./log/',
                        help='Input data path.')
    # parser.add_argument('--epochs', type = int, default = 1,
    #                     metavar = 'N', help = 'number of epochs to train')
    parser.add_argument('--lr', type = float, default = 0.0001,
                        metavar = 'FLOAT', help = 'learning rate')
    parser.add_argument('--embed_dim', type = int, default = 128,
                        metavar = 'N', help = 'embedding dimension')
    parser.add_argument('--n_epoch', type=int, default=500,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=10,
                        help='Number of epoch for early stopping')
    parser.add_argument('--weight_decay', type = float, default = 0.00001,
                        metavar = 'FLOAT', help = 'weight decay')
    parser.add_argument('--N', type = int, default = 30000,
                        metavar = 'N', help = 'L0 parameter')
    parser.add_argument('--droprate', type = float, default = 0.2,
                        metavar = 'FLOAT', help = 'dropout-rate')
    parser.add_argument('--batch_size', type = int, default = 256,
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--test_batch_size', type = int, default = 256,
                        metavar = 'N', help = 'input batch size for testing')
    parser.add_argument('--dataset', type = str, default = 'hh',
                        metavar = 'STRING', help = 'dataset')
    parser.add_argument('--rawpath', type=str, default='./SDPred-main/data',
                        metavar='STRING', help='rawpath')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='GPU or CPU')

    args = parser.parse_args()
    return args


    # ----------------------------------------define log information--------------------------------------------------------

# create log information
def create_log_id(dir_path):
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    return log_count


def logging_config(folder=None, name=None,
                   level=logging.DEBUG,
                   console_level=logging.DEBUG,
                   no_console=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".txt")
    print("All logs will be saved to %s" % logpath)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder
    
    
#--------------------------------------------model-------------------------------------------------------
class ConvNCF(nn.Module):
    def __init__(self, drugs_dim, sides_dim, embed_dim, bathsize, dropout):
        super(ConvNCF, self).__init__()

        self.drugs_dim = drugs_dim
        self.sides_dim = sides_dim
        self.batchsize = bathsize
        # self.drug_dim = self.drugs_dim//6
        # self.side_dim = self.sides_dim//2
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.dim = 512
        self.drugs_layer = nn.Linear(self.drugs_dim, self.dim)
        # self.drugs_layer_1 = nn.Linear(512, 256)
        # self.drugs_layer_2 = nn.Linear(256, 128)
        # self.drugs_layer_3 = nn.Linear(512, 256)
        self.drugs_layer_5 = nn.Linear(512, self.embed_dim)
        self.drugs_layer_4 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drugs_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.sides_layer = nn.Linear(self.sides_dim, 1024)
        self.sides_layer_1 = nn.Linear(1024,512)
        self.sides_layer_2 = nn.Linear(512,256)
        self.sides_layer_4 = nn.Linear(256,self.embed_dim)
        self.sides_layer_3 = nn.Linear(self.embed_dim,self.embed_dim)
        self.sides_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        
        self.num_class = 2
        
        
        self.total_layer = nn.Linear((self.embed_dim * 2), self.embed_dim )
        self.total_bn = nn.BatchNorm1d((self.embed_dim * 2), momentum=0.5)
        self.classifier = nn.Linear(self.embed_dim , 1)
        self.con_layer = nn.Linear(self.embed_dim , 1)
        
        # self.total_layer = nn.Linear((self.embed_dim*2), self.embed_dim)
        # self.total_bn = nn.BatchNorm1d((self.embed_dim*2), momentum=0.5)
        # self.classifier = nn.Linear(self.embed_dim , 1)
        # self.con_layer = nn.Linear(self.embed_dim , 1)

    def forward(self, drug_features, side_features, device):

        # x_drugs = F.relu(self.drugs_bn(self.drugs_layer_5(self.drugs_layer_3(self.drugs_layer_2(self.drugs_layer_1(self.drugs_layer(drug_features.to(device))))))), inplace=True)
        
        x_drugs = F.relu(self.drugs_bn(self.drugs_layer_5(self.drugs_layer(drug_features.to(device)))), inplace=True)
        x_drugs = F.dropout(x_drugs, training=self.training, p=self.dropout)
        x_drugs = self.drugs_layer_4(x_drugs)#3750----128

        x_sides = F.relu(self.sides_bn(self.sides_layer_4(self.sides_layer_2(self.sides_layer_1(self.sides_layer(side_features.to(device)))))), inplace=True)
        x_sides = F.dropout(x_sides, training=self.training, p=self.dropout)
        x_sides = self.sides_layer_3(x_sides)#1988----128

  


        
        

        
        total = torch.cat((x_drugs, x_sides), dim=1)
        total = torch.mean(total,dim=1)

        total = F.relu(self.total_layer(total), inplace=True)
        total = F.dropout(total, training=self.training, p=self.dropout)

        classification = self.classifier(total)
        # total = torch.mean(total,dim=1)

        regression = self.con_layer(total)
        # print(self.total)
        return classification.squeeze(), regression.squeeze()


# -----------------------------------------loading  data------------------------------------------

# loading data


def chunkIt(seq, num):
    data = []
    for i in range(0, len(seq), num):
        if i + num > len(seq):
            data.append(seq[i:])
        else:
            data.append(seq[i:i + num])

    return data


def early_stopping(model, epoch, best_epoch, train_auc,best_auc,bad_counter):
    if train_auc > best_auc :
        best_auc = train_auc
        bad_counter = 0
        save_model(model, args.save_dir, epoch, best_epoch)
        best_epoch = epoch
    else:
        bad_counter += 1
    return bad_counter, best_auc,best_epoch


def save_model(model, model_dir, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.system('rd {}'.format(old_model_state_file))
            


def load_model(model, model_dir, best_epoch):
    model_path = os.path.join(model_dir, 'model_epoch{}.pth'.format(best_epoch))
    checkpoint = torch.load(model_path, map_location=get_device(args))

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            k_ = k[7:]  # remove 'module.' of DistributedDataParallel instance
            state_dict[k_] = v
        model.load_state_dict(state_dict)

    model.eval()
    return model


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def get_device(args):
    args.gpu = False
    if torch.cuda.is_available() and args.cuda:
        args.gpu = True
    device = torch.device("cuda:0" if args.gpu else "cpu")
    return device


# -------------------------------------- metrics and evaluation define -------------------------------------------------
def Accuracy_micro(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        count += sum(np.logical_not(np.logical_xor(y_true[i], y_pred[i])))
    return count / y_true.size
 
def compute_mAP(y_true, y_pred):
    AP = average_precision_score(y_true, y_pred)
    return AP


def spearman(y_true, y_pred):
    sp = stats.spearmanr(y_true, y_pred)[0]
    return sp  

    
    
def calc_metrics(y_true1, pred_score1,y_true2, pred_score2):
    y_pred1 = pred_score1
    n= y_true1.shape
    
    precision, recall, thresholds = metrics.precision_recall_curve(y_true1,
                                                                      y_pred1,
                                                                      pos_label=1,
                                                                      sample_weight=None)
    aupr = metrics.auc(recall, precision)
    fpr, tpr, thresholds2 = metrics.roc_curve(y_true1, y_pred1, pos_label=1)
    auc = metrics.auc(fpr,tpr)
    y_pred1[y_pred1 >= 0.6] = 1
    y_pred1[y_pred1 < 0.6] = 0
    acc = accuracy_score(y_true1, y_pred1)
    precision = metrics.precision_score(y_true1, y_pred1, average='micro')
    recall = metrics.recall_score(y_true1, y_pred1, average='micro')
    f1 = f1_score(y_true1, y_pred1, average='micro')
    # map = get_mAP(y_true1, y_pred1, m=5, n=n, digts=3)
    # map = compute_mAP(y_true1, y_pred1)
    # precisions1, recalls1 = precision_recall_at_k(pred_score1, k=1, threshold=0.6)
    
    
    # sp = spearman(y_true1,pred_score1)
    one_label_index = np.nonzero(y_true1)
    sp = spearman(y_true2[one_label_index],pred_score2[one_label_index])
    rmse = sqrt(mean_squared_error(y_true2[one_label_index],pred_score2[one_label_index]))
    mae = mean_absolute_error(y_true2[one_label_index],pred_score2[one_label_index])

    return acc, precision, recall, f1,auc, aupr,sp,rmse,mae
        
    


def evaluate(args, model, test_loader,device):
    model.eval()
    pred_score1 = []
    pred_score2 = []
    y_true1 = []
    y_true2 = []

    f_drug = []
    f_side = []
    for test_drug, test_side, test_ratings in test_loader:
        test_labels = test_ratings.clone().long()
        for k in range(test_ratings.data.size()[0]):
            if test_ratings.data[k] > 0:
                test_labels.data[k] = 1
        f_drug.append(list(test_drug.data.cpu().numpy()))
        f_side.append(list(test_side.data.cpu().numpy()))
        test_u, test_i, test_ratings = test_drug.to(device), test_side.to(device), test_ratings.to(device)
        
        outputs_1,outputs_2 = model(test_drug, test_side,device)
        
        
        pred_score1.append(list(outputs_1.data.cpu().numpy()))
        pred_score2.append(list(outputs_2.data.cpu().numpy()))   
        y_true2.append(list(test_ratings.data.cpu().numpy()))
        y_true1.append(list(test_labels.data.cpu().numpy()))
        
        

    pred_score1 = np.array(sum(pred_score1, []), dtype = np.float32)
    pred_score2 = np.array(sum(pred_score2, []), dtype=np.float32)

    y_true2 = np.array(sum(y_true2, []), dtype = np.float32)
    y_true1 = np.array(sum(y_true1, []), dtype=np.float32)
    acc, precision, recall, f1,auc, aupr,sp,rmse,mae = calc_metrics(y_true1, pred_score1,y_true2, pred_score2)
    return acc, precision, recall, f1,auc, aupr,sp,rmse,mae
# ----------------------------------------------------------------------------------------------#

def Extract_positive_negative_samples(DAL, addition_negative_number='all'):
    k = 0
    
    interaction_target = np.zeros((DAL.shape[0]*DAL.shape[1], 3)).astype(int)
    
    for i in range(DAL.shape[0]):
        for j in range(DAL.shape[1]):
            interaction_target[k, 0] = i
            interaction_target[k, 1] = j
            interaction_target[k, 2] = DAL[i, j]
            k = k + 1
    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]
    
    
    number_positive = len(np.nonzero(data_shuffle[:, 2])[0])
    
    
    
    final_positive_sample = data_shuffle[interaction_target.shape[0] - number_positive::]
    negative_sample = data_shuffle[0:interaction_target.shape[0] - number_positive]
    
    
    a = np.arange(interaction_target.shape[0] - number_positive)
    a = list(a)
    
    
    if addition_negative_number == 'all':
        b = random.sample(a, (interaction_target.shape[0] - number_positive))
    else:
        b = random.sample(a, (1 + addition_negative_number) * number_positive)
    final_negtive_sample = negative_sample[b[0:number_positive], :]
    addition_negative_sample = negative_sample[b[number_positive::], :]
    return addition_negative_sample, final_positive_sample, final_negtive_sample

# -----------------------------------   train model  -------------------------------------------------------------------

def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set log file
    log_save_id = create_log_id(args.log_dir)
    logging_config(folder=args.log_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    # torch.backends.cudnn.benchmark = True
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # use_cuda = False
    # if torch.cuda.is_available():
    #     use_cuda = True
    # device = torch.device("cuda" if use_cuda else "cpu")



    # initialize data
    gii = open('./SDPred-main/data/drug_side.pkl', 'rb')
    drug_side = pickle.load(gii)
    gii.close()
    
    
    
    #drug features
    gii = open('./SDPred-main/data/Drug_word2vec.pkl', 'rb')
    Drug_word2vec = pickle.load(gii)
    gii.close()
    drug_Tfeature_one = cosine_similarity(Drug_word2vec)
    
    gii = open('./SDPred-main/data/gin_infomax_drug750.npy', 'rb')
    infomax = np.load(gii)
    gii.close()
    drug_Tfeature_two = cosine_similarity(infomax[:750])
    # drug_Tfeature_two = infomax[:750]
    print(drug_Tfeature_two.shape)
    
    
    gii = open('./SDPred-main/data/gin_context_drug750.npy', 'rb')
    context = np.load(gii)
    gii.close()
    drug_Tfeature_three = cosine_similarity(context[:750])
    # drug_Tfeature_three = context[:750]
    print(drug_Tfeature_three.shape)
    

    

    gii = open('./SDPred-main/data/gin_edgepred_drug750.npy', 'rb')
    edge = np.load(gii)
    gii.close()
    drug_Tfeature_four = cosine_similarity(edge[:750])
    # drug_Tfeature_four = edge[:750]
    print(drug_Tfeature_four.shape)

    gii = open('./SDPred-main/data/drugs_fp_750.pkl', 'rb')
    drug_fp = pickle.load(gii)
    gii.close()
    drug_Tfeature_five = cosine_similarity(drug_fp)
    # drug_Tfeature_five = drug_fp
    print(drug_Tfeature_five.shape)
    

    gii = open('./SDPred-main/data/Text_similarity_six.pkl', 'rb')
    drug_Tfeature_six = pickle.load(gii)
    gii.close()
    print(drug_Tfeature_six.shape)
    
    
    
    #side features
    gii = open('./SDPred-main/data/ADR-994.pkl', 'rb')
    adr = pickle.load(gii)
    adr = np.array(adr)
    gii.close()
    print(adr.shape)
    # gii = open('./SDPred-main/data/glove_wordEmbedding.pkl', 'rb')
    # glove_word = pickle.load(gii)
    # gii.close()
    # side_glove_sim = cosine_similarity(glove_word)
    # print(side_glove_sim.shape)
    
    # gii = open('./SDPred-main/data/side_effect_semantic.pkl', 'rb')
    # effect_side_semantic = pickle.load(gii)
    # gii.close()
    # print(effect_side_semantic.shape)


    drug_features, side_features = [], []
    drug_features.append(drug_Tfeature_one)
    drug_features.append(drug_Tfeature_two)
    drug_features.append(drug_Tfeature_three)
    drug_features.append(drug_Tfeature_four)
    drug_features.append(drug_Tfeature_five)
    drug_features.append(drug_Tfeature_six)
 
    side_features.append(adr)
    # side_features.append(effect_side_semantic)
    # side_features.append(side_glove_sim)


    addition_negative_sample, final_positive_sample, final_negative_sample = Extract_positive_negative_samples(drug_side, addition_negative_number='all')
    final_sample = np.vstack((final_positive_sample, final_negative_sample))#74774,3
    X = final_sample[:, 0::]
    final_target = final_sample[:, final_sample.shape[1] - 1]
    y = final_target
    data = []
    data_x = []
    data_y = []

    data_neg_x = []
    data_neg_y = []
    data_neg = []
    for i in range(addition_negative_sample.shape[0]):
        data_neg_x.append((addition_negative_sample[i, 0], addition_negative_sample[i, 1]))
        data_neg_y.append((int(float(addition_negative_sample[i, 2]))))
        data_neg.append((addition_negative_sample[i, 0], addition_negative_sample[i, 1], addition_negative_sample[i, 2]))  
    for i in range(X.shape[0]):
        data_x.append((X[i, 0], X[i, 1]))
        data_y.append((int(float(X[i, 2]))))
        data.append((X[i, 0], X[i, 1], X[i, 2]))

    drug_features_matrix = drug_features[0]
    for i in range(1, len(drug_features)):
        drug_features_matrix = np.hstack((drug_features_matrix, drug_features[i]))
    side_features_matrix = side_features[0]
    for i in range(1, len(side_features)):
        side_features_matrix = np.hstack((side_features_matrix, side_features[i]))
        
        
        
    train_graph = None
        # ====================   training    ====================
    # train model
    # use 10-fold cross validation
    all_precision_list = []
    all_recall_list = []
    all_auc_list = []
    all_aupr_list = []
    all_acc_list = []
    all_Map_list = []
    all_sp_list = []
    all_rmse_list = []
    all_mae_list = []
    all_f1_list = []
    start_t = time()
    kfold = StratifiedKFold(5, random_state=2023, shuffle=True)
    for idx, (train_index, test_index) in enumerate(kfold.split(data_x,data_y)):
        # if idx > 0:
        #     break
        folder = idx + 1
        print(f"***********fold-{folder}***********")
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        c_x_train = data_x[train_index]
        c_y_train = data_y[train_index]
        c_x_test = data_x[test_index]
        c_y_test = data_y[test_index]
        
        
        drug_test = drug_features_matrix[c_x_test[:, 0]]
        side_test = side_features_matrix[c_x_test[:, 1]]
        drug_train = drug_features_matrix[c_x_train[:, 0]]
        side_train = side_features_matrix[c_x_train[:, 1]]
        trainset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_train), torch.FloatTensor(side_train),torch.FloatTensor(c_y_train))
        testset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_test), torch.FloatTensor(side_test),torch.FloatTensor(c_y_test))
        _train = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,num_workers=2, pin_memory=True)
        _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True,num_workers=2, pin_memory=True)
        
        
        
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        use_cuda = False
        if torch.cuda.is_available():
            use_cuda = True
        device = torch.device("cuda" if use_cuda else "cpu")
        print(device)
        
        model = ConvNCF(4500,768, args.embed_dim, args.batch_size,args.droprate)
        model.to(device)
        # logging.info(model)

        bad_counter = 0
        best_auc = 0
        best_aupr = 0
        best_epoch = 0
        avg_loss = 0.0
        Regression_criterion = nn.MSELoss()
        Classification_criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        AUC_mn = 0
        AUPR_mn = 0
    
        rms_mn = 100000
        mae_mn = 100000
        endure_count = 0

        time0 = time()
        for epoch in range(1, args.n_epoch + 1):
            model.train()
            avg_loss = 0.0
            out_list1 = np.empty((0))
            out_list2 = np.empty((0))
            y1 = np.empty((0))
            y2 = np.empty((0))
            for i,data in enumerate(_train,0):
                batch_drug, batch_side, batch_ratings = data
                batch_labels = batch_ratings.clone().float()
                for k in range(batch_ratings.data.size()[0]):
                    if batch_ratings.data[k] > 0:
                        batch_labels.data[k] = 1
                optimizer.zero_grad()
                one_label_index = np.nonzero(batch_labels.data.numpy())
                out1, out2 = model(batch_drug, batch_side, device)
                # out11 = torch.sigmoid(out1)
                # out2 = out2*5
                loss1 = Classification_criterion(out1, batch_labels.to(device))
                loss2 = Regression_criterion(out2[one_label_index], batch_ratings[one_label_index].to(device))
                # print(loss1,loss2)
                total_loss = loss1 * loss2
                # optimizer.zero_grad()
                total_loss.backward(retain_graph = True)
                optimizer.step()
                # optimizer.zero_grad()
                out_list1 = np.concatenate((out_list1, out1.cpu().detach().numpy()), axis=0)
                out_list2 = np.concatenate((out_list2, out2.cpu().detach().numpy()), axis=0)
                y1 = np.concatenate((y1, batch_labels.cpu().detach().numpy()), axis=0)
                y2 = np.concatenate((y2, batch_ratings.cpu().detach().numpy()), axis=0)
                avg_loss += total_loss.item()
            # print(out_list1)
            # print(y1)
            # print(out_list1.shape,y1.shape)
            train_acc, train_precision, train_recall, train_f1,train_auc, train_aupr,train_sp,train_mae,train_rmse = calc_metrics(y1,out_list1,y2,out_list2)
            logging.info('ADR Training: Folder:{}| Epoch:{} | loss:{:.4f}| ACC:{:.4f}| AUC:{:.4f} | AUPR:{:.4f}|Precision:{:.4f}|SP:{:.4f}| MAE:{:.4f}| RMSE:{:.4f}'.format(folder, epoch,avg_loss,train_acc,train_auc,train_aupr,train_precision,train_sp,train_mae,train_rmse))
            
            
            bad_counter, best_auc, best_epoch = early_stopping(model, epoch, best_epoch, train_auc,best_auc,bad_counter)
            if bad_counter >= args.stopping_steps or epoch == args.n_epoch:
                model = load_model(model, args.save_dir, best_epoch)
                acc, precision, recall, f1,auc, aupr,sp,rmse,mae = evaluate(args, model, _test,device)
                logging.info('Final ADR Evaluation: Best_epoch {} |ACC:{:.4f}| AUC:{:.4f} | AUPR:{:.4f}|Precision:{:.4f}|Recall:{:.4f}|SP:{:.4f}|MAE:{:.4f}| RMSE:{:.4f}'.format(best_epoch, acc, auc,aupr,precision,recall,sp, mae, rmse))
                all_precision_list.append(precision)
                all_recall_list.append(recall)
                all_aupr_list.append(aupr)
                all_acc_list.append(acc)
                all_auc_list.append(auc)
                all_sp_list.append(sp)
                all_mae_list.append(mae)
                all_rmse_list.append(rmse)
                break

    mean_acc = np.mean(all_acc_list)
    mean_precision = np.mean(all_precision_list)
    mean_recall = np.mean(all_recall_list)
    mean_aupr = np.mean(all_aupr_list)
    mean_auc = np.mean(all_auc_list)
    mean_f1 = np.mean(all_f1_list)
    mean_mae = np.mean(all_mae_list)
    mean_sp = np.mean(all_sp_list)
    mean_rmse = np.mean(all_rmse_list)
    logging.info('10-fold cross validation DDI Mean Evaluation: Total Time {:.1f}s |ACC:{:.4f}|Precision:{:.4f}|F1:{:.4f}|AUC:{:.4f}| AUPR {:.4f}|SP:{:.4f}|MAE{:.4f}|RMSE:{:.4f}'.format(time()-start_t,  mean_acc, mean_precision, mean_f1,mean_auc,mean_aupr,mean_sp,mean_mae,mean_rmse))


if __name__ == '__main__':
    args = parse_args()
    train(args)