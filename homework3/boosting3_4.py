#!/usr/bin/python
import numpy as np
import math
import random

def read_data(name):
    datas = []
    labels = []
    f = open(name)
    for line in f.readlines():
        arr = list(map(lambda x:int(x),filter(lambda x:x!="",line.split(","))))
        datas.append(arr[1:])
        labels.append(arr[0])
    return np.array(datas),np.array(labels)

class boosting_decision_tree:
    def __init__(self, train_data, train_label,node = True):
        self.train_data = train_data
        self.train_label = train_label
        #print(train_data.shape)
        self.length = train_data.shape[0]
        self.width = train_data.shape[1]
        self.index = -1
        self.wrong_index = []
        self.color = -1
        self.left = None
        self.right = None
        self.loss = 1.0
        if node:
            self.init_tree()
        else:
            self.vote()
    
    def vote(self):
        list_label = list(self.train_label)
        if list_label.count(1) > list_label.count(-1): self.color = 1
        else: self.color = -1
        self.wrong_index = np.argwhere(self.train_label != self.color)

    def predict(self,data):
        #print(data,self.index)
        if data[self.index] == 0:
            return self.left.color
        if data[self.index] == 1:
            return self.right.color

    def split_data(self,attr):
        label_0 = self.train_label[self.train_data[:,attr] == 0]
        data_0 = self.train_data[self.train_data[:,attr] == 0]
        label_1 = self.train_label[self.train_data[:,attr] == 1]
        data_1 = self.train_data[self.train_data[:,attr] == 1]
        return label_0,data_0,label_1,data_1

    def init_tree(self):
        for i in range(self.width):
            leaf_label_0,leaf_data_0,leaf_label_1,leaf_data_1 = self.split_data(i)
            tmp_left = boosting_decision_tree(leaf_data_0, leaf_label_0, False)
            tmp_right = boosting_decision_tree(leaf_data_1, leaf_label_1, False)
            tmp_loss = 1.0 * (len(tmp_left.wrong_index) + len(tmp_right.wrong_index)) / self.length
            if self.loss > tmp_loss:
                self.index = i
                self.left = tmp_left
                self.right = tmp_right
                self.loss = tmp_loss

def main():
    training_data,training_label = read_data("heart_train.data")
    test_data,test_label = read_data("heart_test.data")
    le = len(training_data)
    training_label[training_label == 0] = -1 
    test_label[test_label == 0] = -1 
    forest = []
    for i in range(20):
        index = []
        for j in range(le):
            index.append(random.randint(0,le-1))
        index = np.unique(index)
        bdt = boosting_decision_tree(training_data[index], training_label[index], True)
        forest.append(bdt)
    for tree in forest:
        print(tree.loss)
    predict_label = [0 for i in range(len(test_data))]
    for i in range(len(forest)):
        tree = forest[i]
        for j in range(len(test_data)):
            predict_label[j] += tree.predict(test_data[j])
    print(predict_label)
    predict_label2 = [1 if predict_label[i] > 0 else -1 for i in range(len(test_data))]
    print(predict_label2)
    print(test_label)
    print("accuracy:",1.0 - 1.0 * len(test_label[predict_label2 != test_label]) / len(test_label))

if __name__ == "__main__":
    main()
