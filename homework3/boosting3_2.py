#!/usr/bin/python
import numpy as np
import math

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
    def __init__(self, train_data, train_label, i):
        self.train_data = train_data
        self.train_label = train_label
        #print(train_data.shape)
        self.length = train_data.shape[0]
        self.width = train_data.shape[1]
        self.index = i
        self.wrong_index = []
        self.color = -1
        self.left = None
        self.right = None
        self.loss = 1.0
        self.predict_label = []
    
    def vote(self):
        list_label = list(self.train_label)
        if list_label.count(1) > list_label.count(-1): self.color = 1
        else: self.color = -1
        self.wrong_index = np.argwhere(self.train_label != self.color)

    def predict(self,data):
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
        leaf_label_0,leaf_data_0,leaf_label_1,leaf_data_1 = self.split_data(self.index)
        self.left = boosting_decision_tree(leaf_data_0, leaf_label_0, 0)
        self.left.vote()
        self.right = boosting_decision_tree(leaf_data_1, leaf_label_1, 0)
        self.right.vote()
        self.loss = 1.0 * (len(self.left.wrong_index) + len(self.right.wrong_index)) / self.length

def main():
    training_data,training_label = read_data("heart_train.data")
    test_data,test_label = read_data("heart_test.data")
    training_label[training_label == 0] = -1 
    test_label[test_label == 0] = -1 
    forest = []
    for i in range(training_data.shape[1]):
        bdt = boosting_decision_tree(training_data, training_label, i)
        bdt.init_tree()
        forest.append(bdt)
    for tree in forest:
        print(tree.loss)
    for i in range(len(forest)):
        tree = forest[i]
        for j in training_data:
            tree.predict_label.append(tree.predict(j))
        tree.predict_label = np.array(tree.predict_label)
    #    print("loss:",1.0 * len(tree.train_label[tree.predict_label != training_label]) / len(tree.train_label))
    a_arr = [1.0 for i in range(len(forest))]
    tree_arr = []
    step_size = 0.1
    min_loss = 1.0
    for rounds in range(10):
        for ite in range(len(a_arr)):
            #print("a_arr:",a_arr)
            tree = forest[ite]
            sum_1 = np.zeros(len(tree.predict_label[tree.predict_label == training_label]))
            sum_2 = np.zeros(len(tree.predict_label[tree.predict_label != training_label]))
            for i in range(len(forest)):
                if i == ite: continue
                #print("lables:",forest[i].predict_label[tree.predict_label == training_label])
                sum_1 += a_arr[i] * forest[i].predict_label[tree.predict_label == training_label]
                sum_2 += a_arr[i] * forest[i].predict_label[tree.predict_label != training_label]
            #print("sum_1:",sum_1)
            #print("sum_2:",sum_2)
            at_derivative = math.log(np.sum(np.exp(- training_label[training_label == tree.predict_label] * sum_1)) / np.sum(np.exp(- training_label[training_label != tree.predict_label] * sum_2))) / 2.0
            #print(np.sum(np.exp(- training_label[training_label == tree.predict_label] * sum_1)))
            #print(np.sum(np.exp(- training_label[training_label != tree.predict_label] * sum_2)))
            print("at_derivative:",at_derivative)
            a_arr[ite] -= step_size * at_derivative
        print("a_arr:",a_arr)
        predict_label = [0.0 for i in range(len(test_data))]
        for i in range(len(test_data)):
            for j in range(len(forest)):
                predict_label[i] += a_arr[j] * forest[j].predict(test_data[i])
        predict_label2 = [-1 if predict_label[i] < 0 else 1 for i in range(len(predict_label))]
        print("predict:",predict_label2)
        print("test:",test_label)
        wrong_index = np.argwhere(predict_label2 != test_label)
        #print("wrong_index:",wrong_index)
        loss = 1.0 * len(wrong_index) / len(test_label)
        print("accuracy:",1 - loss)
        if loss < min_loss:
            min_loss = loss
        loss2 = np.sum(np.exp(test_label * np.array(predict_label)))
        print("loss2:",loss2)
    print("min_loss:",min_loss)
    
if __name__ == "__main__":
    main()
