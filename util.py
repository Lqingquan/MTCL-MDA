import numpy as np
import torch


def encoder_miRNA_disease_ID(name):
    adj = np.loadtxt("data/{}/interaction.txt".format(name), dtype=int, delimiter=" ")
    n = []
    with open('dataset/D2R_3/D2R_3.inter', 'w') as file:
        for i in range(len(adj[0])):
            for j in range(len(adj)):
                if adj[j][i] == 1:
                    if j not in n:
                        n.append(j)
                    file.write(str(i + 1) + '\t' + str(n.index(j) + 1) + '\n')


def load_data(name):
    dis_embedding = np.loadtxt('data/HMDDv2/disease_embedding.txt', dtype=float, delimiter=" ")
    rna_embedding = np.loadtxt('data/HMDDv2/miRNA_embedding.txt', dtype=float, delimiter=" ")
    adj = np.loadtxt('data/HMDDv2interaction.txt', dtype=float, delimiter=" ")
    pos_data = []
    neg_data = []

    for i in range(len(adj)):
        for j in range(len(adj[0])):
            if adj[i][j] == 1:
                pos_data.append([rna_embedding[i], dis_embedding[j]])
            else:
                neg_data.append([rna_embedding[i], dis_embedding[j]])
    return pos_data, neg_data


def split_data(pos_data, neg_data):
    n_data = len(pos_data)
    neg_list = []
    while n_data:
        index = np.random.randint(len(neg_data))
        neg_list.append(neg_data[index])
        del neg_data[index]
        n_data = n_data - 1
    pos_list = pos_data
    return pos_list, neg_list


def deal_embedding(list):
    result = []
    for i in range(len(list)):
        result.append(np.concatenate([list[i][0], list[i][1]], axis=0))
    return result


def split_train_test(pos_list, neg_list, fold, times):
    train_data, train_label = [], []
    test_data, test_label = [], []
    for i in range(len(pos_list)):
        if i % fold != times:
            train_data.append(pos_list[i])
            train_label.append(1)
            train_data.append(neg_list[i])      # 采集与正样本数量相同的负样本
            train_label.append(0)
        else:
            test_data.append(pos_list[i])
            test_label.append(1)
            test_data.append(neg_list[i])
            test_label.append(0)

    train_data, train_label, test_data, test_label = torch.tensor(train_data), torch.tensor(train_label), torch.tensor(test_data), torch.tensor(test_label)
    return train_data.to(torch.float32), train_label.to(torch.float32), test_data.to(torch.float32), test_label.to(torch.float32)


def calculate_metrics(y_true, y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    precision = TP / (TP + FP)
    specificity = TN / (TN + FP)
    F1_score = 2*(precision*sensitivity)/(precision+sensitivity)
    return accuracy, sensitivity, precision, specificity, F1_score





