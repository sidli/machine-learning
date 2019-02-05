#!/usr/bin/python
import numpy as np
import math

def loadDataSet(filename):
    dataArr = []
    labelArr = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = list(filter(lambda x: x != "",line.strip().split(',')))
        labelArr.append(lineArr[0])
        dataArr.append(lineArr[1:])
    dataArr = np.array(dataArr)
    labelArr = np.array(labelArr)
    return dataArr,labelArr

def Calculate_accuracy(y1, y2):
    if len(y1) != len(y2):
        print(len(y1),len(y2)," Different Length Error")
        sys.exit(1)
    sumOfSquares = 0.0 
    success_index = [i for i in range(len(y1)) if y1[i] == y2[i]]
    for i in range(len(y1)):
        sumOfSquares += (y1[i] == y2[i])
    return (sumOfSquares / len(y1)), success_index

class tree():
    def __init__(self,datas,labels,column_names,height = 0):
        self.datas = datas
        self.labels = labels
        self.column_names = column_names
        self.length = len(self.datas)
        self.subtrees = []
        self.classfier = {}
        self.select_column = ""
        self.color = "None"
        self.accuracy = 0.0
        self.height = height

    def statistic_column(self,column):
        P_y_dic = {}
        for key in np.unique(column):
            p = column[column == key].size / len(column)
            P_y_dic[key] = p
        return P_y_dic

    def H(self,column):
        dic = self.statistic_column(column)
        ps = np.array(list(dic.values()))
        return -np.sum(ps * np.log2(ps))
    
    def vote(self,column):
        votes = []
        keys = np.unique(column)
        for key in keys:
            vote = column[column == key].size / len(column)
            votes.append(vote)
        print(votes)
        color = keys[votes.index(max(votes))]
        accuracy = column[column == color].size / len(column)
        return color, accuracy

    def train(self):
        #print(self.datas.shape)
        print("-" * 50)
        print("height:",self.height," data.shape:",self.datas.shape)
        self.color,self.accuracy = self.vote(self.labels)
        print("color:",self.color," accuracy:",self.accuracy)
        if self.datas.shape[0] in [0,1] or self.datas.shape[1] == 0 or self.accuracy == 1.0:
            return "finished"
        dic_y = self.statistic_column(self.labels)
        H_Y = self.H(self.labels)
        
        H_Xs = [0.0 for i in range(self.datas.shape[1])]
        for column in range(self.datas.shape[1]):
            dic_x = self.statistic_column(self.datas[:,column])
            for key in dic_x.keys():
                #print(self.datas[:,column] == key)
                H_Xs[column] += dic_x[key] * self.H(self.labels[self.datas[:,column] == key])
        #print(H_Xs)
        select = np.argmax(H_Y - H_Xs)
        self.select_column = self.column_names[select]
        print("select index:",select," column:",self.select_column)
        #print("keys:",np.unique(self.datas[:,select]))
        keys = np.unique(self.datas[:,select])
        for i,key in zip(range(len(keys)), keys):
            self.classfier[key] = i
        for key in self.classfier.keys():
            #print(self.datas[self.datas[:,select] == key])
            subdatas = np.delete(self.datas[self.datas[:,select] == key], select, axis = 1)
            sublabels = self.labels[self.datas[:,select] == key]
            subcolunms = np.delete(self.column_names, select)
            self.subtrees.append(tree(subdatas, sublabels, subcolunms, self.height + 1))
        print("childrens number:",len(self.subtrees))

def main():
    train_data,train_label = loadDataSet("mush_train.data")
    test_data,test_label = loadDataSet("mush_test.data")
    column_names = ["cap-shape","cap-surface","cap-color","bruises","odor","gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type","veil-color","ring-number","ring-type","spore-print-color","population","habitat"]
    root = tree(train_data,train_label,np.array(column_names))
    #for i in range(10)
    trees = [root]
    iteration = 0
    while(len(trees) != 0):
        tree_node = trees.pop(0)
        tree_node.train()
        trees += tree_node.subtrees
        iteration += 1
    print("iteration:",iteration)

    predict_label = []
    for item in test_data:
        tree_node = root
        while(tree_node.subtrees != []):
            key = item[column_names.index(tree_node.select_column)]
            tree_node = tree_node.subtrees[tree_node.classfier[key]]
        predict_label.append(tree_node.color)
    accuracy,indexes = Calculate_accuracy(predict_label,test_label)
    print("accuracy:",accuracy)

if __name__ == "__main__":
    main()
