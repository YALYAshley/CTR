#####模块说明######
'''
根据传入的文件true_label和predict_label来求模型预测的精度、召回率和F1值，另外给出微观和宏观取值。
powered by wangjianing 2019.3.2
'''
import numpy as np
 
def getLabel2idx(labels):
    '''
    获取所有类标
    返回值：label2idx字典，key表示类名称，value表示编号0,1,2...
    '''
    label2idx = dict()
    for i in labels:
        if i not in label2idx:
            label2idx[i] = len(label2idx)
    return label2idx
 
 
def buildConfusionMatrix(predict,gt):
    '''
    针对实际类标和预测类标，生成对应的矩阵。
    矩阵横坐标表示实际的类标，纵坐标表示预测的类标
    矩阵的元素(m1,m2)表示类标m1被预测为m2的个数。
    所有元素的数字的和即为测试集样本数，对角线元素和为被预测正确的个数，其余则为预测错误。
    返回值：返回这个矩阵numpy
    '''
    label2idx = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7}
    confMatrix = np.zeros([len(label2idx),len(label2idx)],dtype=np.int32)
    for i in range(len(gt)):
        true_labels_idx = label2idx[gt[i]]
        predict_labels_idx = label2idx[predict[i]]
        confMatrix[true_labels_idx][predict_labels_idx] += 1
    return confMatrix,label2idx
 
 
 
def calculate_all_prediction(confMatrix):
    '''
    计算总精度：对角线上所有值除以总数
    '''
    total_sum = confMatrix.sum()
    correct_sum = (np.diag(confMatrix)).sum()
    prediction = round(100*float(correct_sum)/float(total_sum),2)
    return prediction
 
def calculate_label_prediction(confMatrix,labelidx):
    '''
    计算某一个类标预测精度：该类被预测正确的数除以该类的总数
    '''
    label_total_sum = confMatrix.sum(axis=0)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    prediction = 0
    if label_total_sum != 0:
        prediction = round(100*float(label_correct_sum)/float(label_total_sum),2)
    return prediction
 
def calculate_label_recall(confMatrix,labelidx):
    '''
    计算某一个类标的召回率：
    '''
    label_total_sum = confMatrix.sum(axis=1)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    recall = 0
    if label_total_sum != 0:
        recall = round(100*float(label_correct_sum)/float(label_total_sum),2)
    return recall
 
def calculate_f1(prediction,recall):
    if (prediction+recall)==0:
        return 0
    return round(2*prediction*recall/(prediction+recall),2)
