import pdb
import argparse
import logging
from logging import debug as DEBUG
from logging import info as INFO
from logging import error as ERROR
logging.basicConfig(format=f"[%(levelname)s %(filename)s line:%(lineno)d] %(message)s", level=logging.INFO)


from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import numpy as np
import matplotlib.pyplot as plt
import cv2

import numpy as np
#import pandas as pd

class Heart():
    def __init__(self, args) -> None:
        self.x = []
        self.y = []
        self.z = []

    def plot1(self):
        with open("../data/Dbig.txt", "r") as f:
            for line in f:
                numlist = line.split()
                self.x.append(numlist[0])
                self.y.append(numlist[1])
                self.z.append(numlist[2])

        plt.figure(figsize=(5,5))
        for i in range(len(self.x)):
            x = float(self.x[i])
            y = float(self.y[i])
            z = float(self.z[i])
            INFO(f"{x} {y} {z}")
            # pdb.set_trace()
            if(z == 1):
                plt.plot(x, y, 'o', markersize=0.5, color="b")
            else:
                plt.plot(x, y, 'o', markersize=0.5, color="r")
            # if(i>100):break
        plt.axis('equal')
        ax = plt.gca()
        xml = plt.MultipleLocator(0.5)
        yml = plt.MultipleLocator(0.5)
        ax.xaxis.set_major_locator(xml)
        ax.yaxis.set_major_locator(yml)
        plt.xlim((-2, 2))
        plt.ylim((-2, 2))
        plt.savefig('../output/fig1.png')
        plt.close()
    
    def plot2(self):
        with open("../data/Dbig.txt", "r") as f:
            for line in f:
                numlist = line.split()
                self.x.append(numlist[0])
                self.y.append(numlist[1])
                self.z.append(numlist[2])

        plt.figure(figsize=(10,10))
        for i in range(len(self.x)):
            x = float(self.x[i])
            y = float(self.y[i])
            z = float(self.z[i])
            INFO(f"{x} {y} {z}")
            # pdb.set_trace()
            plt.plot(x, y, 'o', markersize=0.5, color="b")
            # if(i>100):break
        plt.axis('equal')
        ax = plt.gca()
        xml = plt.MultipleLocator(0.5)
        yml = plt.MultipleLocator(0.5)
        ax.xaxis.set_major_locator(xml)
        ax.yaxis.set_major_locator(yml)
        plt.xlim((-2, 2))
        plt.ylim((-2, 2))
        plt.savefig('../output/fig2.png')
        plt.close()
        
    def plot3(self, img_name, refine_img_name):
        img = cv2.imread(img_name)
        INFO(f"{type(img)}")
        INFO(f"{img.shape}")
        for i in range(img.shape[0]):
        # for i in range(550, 600):
            # if(i == 300): break
            for j in range(img.shape[1]):
                INFO(f"{i} {j} {img[i][j]}")
                if(img[i][j][0] < 128 and img[i][j][1] > 128 and img[i][j][2] > 128):
                    img[i][j][0] = 0
                    img[i][j][1] = 0
                    img[i][j][2] = 255
                elif(img[i][j][0] >200 and img[i][j][1] < 50 and img[i][j][2] < 50):
                    img[i][j][0] = 238
                    img[i][j][1] = 130
                    img[i][j][2] = 238
        cv2.imwrite(refine_img_name, img)                

    def plot4(self):
        for i in range(4):
            img_name = "../7-"+str(i+2)+".png"
            refine_img_name = "../output/"+"7-"+str(i+2)+".png"
            self.plot3(img_name, refine_img_name)


    def plot5(self):
        x = [32,128,512,2048,8192]
        # y = [0.191, 0.098, 0.047, 0.027, 0.019]
        y = [0.191, 0.098, 0.047, 0.027, 0.019]
        plt.plot(x, y, color="r")
        plt.savefig('../output/7-3-1.png')
        plt.close()
    
    def plot6(self):
        x = [32,128,512,2048,8192]
        # y = [0.191, 0.098, 0.047, 0.027, 0.019]
        y = [0.135, 0.12, 0.057, 0.027, 0.011]
        plt.plot(x, y, color="r")
        plt.savefig('../output/7-3-2.png')
        plt.close()

    def Q4(self):
        x = np.array([0, 1, 2])
        y = x**3
        poly = lagrange(x, y)
        INFO(f"{poly(0)} {poly(1)} {poly(2)} {poly(3)} {poly(10)}")
        # INFO(f"{np.sin(10)}")
        a = 0
        b = 1
        testsetnum = 20
        x = np.random.uniform(a, b, 100+testsetnum)
        y = np.sin(x)
        x_train = x[:100]
        y_train = y[:100]
        x_test = x[100:]
        y_test = y[100:]
        INFO(f"{x_train.shape} {y_train.shape}")
        poly = lagrange(x_train, y_train)
        # INFO(f"{poly}")
        INFO(f"5: {poly(5)} {np.sin(5)}")
        pred_train = poly(x_train)
        pred_test = poly(x_test)
        plt.plot(x_train, y_train, 'o', markersize=0.5, color="b")
        # plt.plot(x_train, pred_train, 'o', markersize=0.5, color="r")
        plt.savefig('../output/9-1.png')
        plt.close()
        # plt.plot(x_test, y_test, 'o', markersize=0.5, color="b")
        # plt.plot(x_test, pred_test, 'o', markersize=0.5, color="r")
        # plt.savefig('../output/9-2.png')
        # plt.close()

        train_err = y_train - pred_train
        test_err = y_test - pred_test
        INFO(f"train_err {np.linalg.norm(train_err)/100}")
        INFO(f"test_err {np.linalg.norm(test_err)/testsetnum}")
        INFO(f"{y_train[0:10]}")
        INFO(f"{pred_train[0:10]}")

        for e in (0.001, 0.01, 0.1, 1, 10, 100):
            y_noise = y_train + np.random.normal(0, e, 100)
            poly_noise = lagrange(x_train, y_noise)
            pred_train_noise = poly_noise(x_train)
            pred_test_noise = poly_noise(x_test)
            plt.plot(x_train, y_train, 'o', markersize=0.5, color="b")
            # plt.plot(x_train, pred_train, 'o', markersize=0.5, color="r")
            # plt.savefig('../output/9-1.png')
            # plt.close()
            # plt.plot(x_test, y_test, 'o', markersize=0.5, color="b")
            # plt.plot(x_test, pred_test, 'o', markersize=0.5, color="r")
            # plt.savefig('../output/9-2.png')
            # plt.close()
            train_err_noise = y_train - pred_train_noise
            test_err_noise = y_test - pred_test_noise
            INFO(f"{e} train_err_noise {np.linalg.norm(train_err_noise)/100}")
            INFO(f"test_err_noise {np.linalg.norm(test_err_noise)/testsetnum}")
       
class DT_Node:
    def __init__(self, f_vector, threshold, l_leaf, r_leaf, label):
        self.label = label
        self.f_vector = f_vector
        self.threshold = threshold
        self.l_leaf = l_leaf
        self.r_leaf = r_leaf


    def infogain(self, data, pred, f_vector, threshold):
        pred_1 = pred[data[:, f_vector] >= threshold]
        pred_2 = pred[data[:, f_vector] < threshold]
        info_gain =  entropy(pred) - len(pred_1) / len(pred) * entropy(pred_1) - len(pred_2) / len(pred) * entropy(pred_2)
        if len(pred_1) / len(pred)==0 or len(pred_1) / len(pred)==1:
            generate_entr = 0
        else:
            generate_entr = -len(pred_1) / len(pred) * np.log2(len(pred_1) / len(pred)) - len(pred_2) / len(pred) * np.log2(len(pred_2) / len(pred))

        if generate_entr == 0:
            return None 
        else:
            return info_gain / generate_entr

    def find_best_split(self, data, pred):
        top_f = None
        top_thresh = None
        maximum = 0
        for f_vector in range(data.shape[1]):
            for i in np.unique(data[:, f_vector]):
                ig_ratio = infogain(data, pred, f_vector, i)
                if ig_ratio > maximum:
                    maximum = ig_ratio
                    top_f = f_vector
                    top_thresh = i
        return (top_f, top_thresh, maximum)

    def entropy(self, pred):
        if(pred.size != 0):
            prob_1 = np.mean(pred)
            if prob_1 == 0 or prob_1 == 1:
                return 0
            else:
                return -prob_1 * np.log2(prob_1) - (1 - prob_1) * np.log2(1 - prob_1)

    def generate_train_test(self, data, pred, f_vector, threshold):
        return data[data[:, f_vector] >= threshold], pred[data[:, f_vector] >= threshold], data[data[:, f_vector] < threshold], pred[data[:, f_vector] < threshold]

    def tree(self, data, pred):
        f_vector, threshold, ig_ratio = find_best_split(data, pred) 
        if len(pred)==0 or ig_ratio ==0:
            if np.mean(pred) >= 0.5:
                label = 1
            else:
                label = 0
            return Node(f_vector = None, threshold = None, l_leaf = None, r_leaf = None, label = label)
        else:
            data_1, pred_1, data_2, pred_2 = generate_train_test(data, pred, f_vector, threshold)
            return Node(f_vector = f_vector, threshold = threshold, l_leaf = tree(data_1, pred_1), r_leaf = tree(data_2, pred_2), label = None)
    
    def print_node(self, tree):
        if tree.label is not None:
            return 1
        result = print_node(tree.l_leaf) + print_node(tree.r_leaf) + 1
        return result

    def print_error(self, tree, data, pred):
        error = 1 - np.mean(np.array([predict(tree, x) for x in data]) == pred)
        return error

    def predict(self, tree, x):
        if tree.label is not None:
            return tree.label
        if x[tree.f_vector] >= tree.threshold:
            return predict(tree.l_leaf, x)
        else:
            return predict(tree.r_leaf, x)

    def read_file(self, file_path):
        data = np.loadtxt(file_path, delimiter=' ')
        data = data[:, :-1]
        pred = data[:, -1]
        return data, pred 

    def test(self, trainfile, testfile):
        result = np.mean(np.array([predict(tree(read_file(trainfile), read_file(testfile)), x) for x in X_test]) == y_test)
        return result


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--wxy", type=int, help='wxy')
   

    args = parser.parse_args()

    # print(f"{args}")
    dt_node = DT_Node(args)
    heart = Heart(args)
    # heart.plot1()
    # heart.plot2()
    # heart.plot3()
    # heart.plot4()
    # heart.plot5()
    # heart.plot6()
    heart.Q4()
