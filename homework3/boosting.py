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
    def __init__(self, train_data, train_label, weight, nodes_num):
        self.train_data = train_data
        self.train_label = train_label
        #print(train_data.shape)
        self.length = train_data.shape[0]
        self.width = train_data.shape[1]
        self.nodes_num = nodes_num
        self.wrong_index = []
        self.weight = weight
        self.color = -1
        self.min_weight = 1.0e+8
        if nodes_num == 0: self.leaf = True
        else: self.leaf = False
        self.vote()
        self.best_decision_tree = []
        self.left_child = None
        self.right_child = None
    
    def vote(self):
        list_label = list(self.train_label)
        if list_label.count(1) > list_label.count(0): self.color = 1
        else: self.color = 0
        if self.leaf == True:
            self.min_weight = np.sum(self.weight[self.train_label != self.color])
            self.wrong_index

    def predict(self,data):
        if self.leaf == True:
            return self.color
        index = self.best_decision_tree[0]
        vector = self.best_decision_tree[2]
        if data[index] == vector[0]:
            return self.left_child.predict(data)
        if data[index] == vector[1]:
            return self.right_child.predict(data)

    def split_data(self,attr,value):
        label_0 = self.train_label[self.train_data[:,attr] == value[0]]
        data_0 = self.train_data[self.train_data[:,attr] == value[0]]
        label_1 = self.train_label[self.train_data[:,attr] == value[1]]
        data_1 = self.train_data[self.train_data[:,attr] == value[1]]
        weight_0 = self.weight[self.train_data[:,attr] == value[0]]
        weight_1 = self.weight[self.train_data[:,attr] == value[1]]
        return label_0,data_0,label_1,data_1,weight_0,weight_1

    def one_leaf_one_sub_tree(self,attr,value):
        sub_label,sub_data,leaf_label,leaf_data,sub_weight,leaf_weight = self.split_data(attr,value)
        subtree = boosting_decision_tree(sub_data, sub_label, sub_weight, self.nodes_num - 1)
        leaf = boosting_decision_tree(leaf_data, leaf_label, leaf_weight, 0)
        return subtree.select_best_tree() + leaf.min_weight, subtree, leaf

    def two_sub_tree(self,attr):
        sub_label_0,sub_data_0,sub_label_1,sub_data_1,sub_weight_0,sub_weight_1 = self.split_data(attr,[0,1])
        subtree_0 = boosting_decision_tree(sub_data_0, sub_label_0, sub_weight_0, (self.nodes_num-1)/2)
        subtree_1 = boosting_decision_tree(sub_data_1, sub_label_1, sub_weight_1, (self.nodes_num-1)/2)
        return subtree_0.select_best_tree() + subtree_1.select_best_tree(), subtree_0, subtree_1

    def two_leaf(self,attr):
        leaf_label_0,leaf_data_0,leaf_label_1,leaf_data_1,leaf_weight_0,leaf_weight_1 = self.split_data(attr,[0,1])
        leaf_0 = boosting_decision_tree(leaf_data_0, leaf_label_0, leaf_weight_0, 0)
        leaf_1 = boosting_decision_tree(leaf_data_1, leaf_label_1, leaf_weight_1, 0)
        return leaf_0.min_weight + leaf_1.min_weight, leaf_0, leaf_1

    def select_best_tree(self):
        for i in range(self.width): #attributes iteration
            #two subtree
            if self.nodes_num >= 3:
                min_weight,node1,node2 = self.two_sub_tree(i)
                if min_weight < self.min_weight:
                    self.min_weight = min_weight
                    self.left_child, self.right_child = node1, node2
                    self.best_decision_tree = [i, "two subtree",[0,1]]
            #one leaf one subtree
            if self.nodes_num >= 2:
                min_weight,node1,node2 = self.one_leaf_one_sub_tree(i,[0,1])
                if min_weight < self.min_weight:
                    self.min_weight = min_weight
                    self.left_child, self.right_child = node1, node2
                    self.best_decision_tree = [i, "one leaf one subtree", [0,1]]
                min_weight,node1,node2 = self.one_leaf_one_sub_tree(i,[1,0])
                if min_weight < self.min_weight:
                    self.min_weight = min_weight
                    self.left_child, self.right_child = node1, node2
                    self.best_decision_tree = [i, "one leaf one subtree", [1,0]]
            #two leafs
            if self.nodes_num == 1:
                min_weight,node1,node2 = self.two_leaf(i)
                if min_weight < self.min_weight:
                    self.min_weight = min_weight
                    self.left_child, self.right_child = node1, node2
                    self.best_decision_tree = [i, "two leaf", [0,1]]
        return self.min_weight

def main():
    training_data,training_label = read_data("heart_train.data")
    weight = np.array([1.0/len(training_data) for i in range(len(training_data))])
    weights = []
    for i in range(6):
        if i < 5:
            weights.append(weight)
        if i == 5:
            weight = np.sum(np.array(weights),axis=0)
        print("weight:", weight)
        bdt = boosting_decision_tree(training_data, training_label, weight, 3)
        bdt.select_best_tree()
        trees = [bdt]
        while(len(trees) != 0):
            tree = trees.pop(0)
            print("decision:",tree.best_decision_tree)
            print("total:",tree.length, "  min_weight:",tree.min_weight, " color:", tree.color, "leaf" if tree.leaf == True else "trunk")
            if tree.left_child != None: trees.append(tree.left_child)
            if tree.right_child != None: trees.append(tree.right_child)
        wrong_index = []
        for j in range(len(training_data)):
            if bdt.predict(training_data[j]) != training_label[j]:
                wrong_index.append(j)
        print("wrong index:", len(wrong_index), " ", wrong_index)
        if i == 5:
            break
        error_rate = bdt.min_weight
        ai = math.log((1.0 - error_rate)/error_rate) / 2.0
        denominator = 2.0 * math.sqrt(error_rate * (1.0 - error_rate))
        print("error_rate:",error_rate, " at:", ai)
        for j in range(len(training_data)):
            if j in wrong_index:
                weight[j] = weight[j] * math.exp(ai) / denominator
            else:
                weight[j] = weight[j] * math.exp(-ai) / denominator
        print("-"*50)

if __name__ == "__main__":
    main()
