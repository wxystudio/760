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
import pandas as pd
import cv2
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc


class HW31():
    def __init__(self, args) -> None:
        self.data = np.loadtxt('../data/D2z.txt')
        INFO(f"data: {self.data.shape}")
    
        # pdb.set_trace()

    def run1(self):
        x1_test = []
        x2_test = []
        for i, a in enumerate(np.arange(-2, 2.1, 0.1)):
            for j, b in enumerate(np.arange(-2, 2.1, 0.1)):
                x1_test.append(a)
                x2_test.append(b)

        x1_test = np.array(x1_test)
        x2_test = np.array(x2_test)
        feature_test = np.vstack((x1_test, x2_test)).T
        INFO(f"x1_test: {x1_test.shape}")
        INFO(f"x2_test: {x2_test.shape}")
        INFO(f"feature_test: {feature_test.shape}")

        pred = self.knn(self.data, feature_test, 1)
        INFO(f"pred: {pred.shape}")
        testset = np.append(feature_test, pred.reshape(1681,1), 1)
        INFO(f"testset: {testset.shape}")
        # pdb.set_trace()
        x1_test_pos = testset[testset[:, 2]==1][:, 0]
        x2_test_pos = testset[testset[:, 2]==1][:, 1]
        x1_test_neg = testset[testset[:, 2]==0][:, 0]
        x2_test_neg = testset[testset[:, 2]==0][:, 1]
        x1_train_pos = []
        x2_train_pos = []
        x1_train_neg = []
        x2_train_neg = []
        for i in range(self.data.shape[0]):
            if(self.data[i][2] == 1):
                x1_train_pos.append(self.data[i][0])
            if(self.data[i][2] == 1):
                x2_train_pos.append(self.data[i][1])
            if(self.data[i][2] == 0):
                x1_train_neg.append(self.data[i][0])
            if(self.data[i][2] == 0):
                x2_train_neg.append(self.data[i][1])
        x1_train_pos = np.array(x1_train_pos)
        x2_train_pos = np.array(x2_train_pos)
        x1_train_neg = np.array(x1_train_neg)
        x2_train_neg = np.array(x2_train_neg)
        INFO(f"x1_train_pos: {x1_train_pos.shape} {x1_train_pos[:10]}")
        # pdb.set_trace()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.plot(x1_test_pos, x2_test_pos, '.', color='green', markersize=2)
        plt.plot(x1_test_neg, x2_test_neg, '.', color='deeppink', markersize=2)
        plt.plot(x1_train_pos, x2_train_pos, 'o', color='green', markersize=2)
        plt.plot(x1_train_neg, x2_train_neg, 'x', color='deeppink', markersize=2)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.savefig("../output/21.png")

    def find(self, train, test_row, num_neighbors):
        distances = list()
        for i in range(train.shape[1]):
            train_row = train[i]
            # INFO(f"train_row: {train_row}")
            dist = np.linalg.norm(test_row - np.array(train_row)[0:2])
            # INFO(f"test_row: {test_row}")
            distances.append((train_row, dist))
            # pdb.set_trace()

        distances.sort(key=lambda x: x[1])
        neighbors = []
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors

    def knn(self, train, test, num_neighbors):
        # INFO(f"train: {type(train)} {train.shape}")
        # pdb.set_trace()
        predictions = []
        for row in test:
            neighbors = self.find(train, row, num_neighbors)
            output_values = [row[-1] for row in neighbors]
            output = max(set(output_values), key=output_values.count)
            predictions.append(output)

        INFO(f"{predictions[:10]}")
        # pdb.set_trace()
        predictions = np.array(predictions)
        return(predictions)


class HW32():
    def __init__(self, args) -> None:
        self.data = None
        with open('../data/emails.csv', encoding='utf-8') as f:
            self.data = np.loadtxt('../data/emails.csv', dtype=str, delimiter=',', skiprows=1)
        INFO(f"data: {self.data.shape} {self.data[:5, :5]}")
        self.data = self.data[:, 1:]
        INFO(f"data: {self.data.shape} {self.data[:5, :5]}")

        self.trainset = self.data[:4000, :]
        self.testset = self.data[4000:, :]
        self.train_fold = []
        self.test_fold = []
        self.train_fold.append(self.data[:1000, :])
        self.train_fold.append(self.data[1000:2000, :])
        self.train_fold.append(self.data[2000:3000, :])
        self.train_fold.append(self.data[3000:4000, :])
        self.train_fold.append(self.data[4000:5000, :])
        self.test_fold.append(self.data[1000:, :])
        self.test_fold.append(np.concatenate((self.data[:1000, :], self.data[2000:, :]), axis=0))
        self.test_fold.append(np.concatenate((self.data[:2000, :], self.data[3000:, :]), axis=0))
        self.test_fold.append(np.concatenate((self.data[:3000, :], self.data[4000:, :]), axis=0))
        self.test_fold.append(self.data[:4000, :])

        INFO(f"self.trainset: {self.trainset.shape}")


    def plot(self):
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(self.trainset.iloc[:, :-1], self.trainset.iloc[:, -1])
        y_pred = neigh.predict(self.testset.iloc[:, :-1])
        fpr, tpr, threshold = roc_curve(self.testset.iloc[:, -1], y_pred)

        lr = LogisticRegression(max_iter=1000)
        lr.fit(self.trainset.iloc[:, :-1], self.trainset.iloc[:,-1])
        predicted_classes = lr.predict(self.testset.iloc[:, :-1])

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', color="violet", label="kNN : AUC = 0.770")
        fpr, tpr, threshold = roc_curve(self.testset.iloc[:, -1], predicted_classes)
        plt.plot(fpr, tpr, 'b', color="purple", label="Logistic Regression : AUC = 0.936")
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.show()

    def run1(self):
        self.k_list = [1,3,5,7,10]
        for i in self.k_list:
            var = 0
            neigh = KNeighborsClassifier(n_neighbors=i)
            neigh.fit(self.train_fold[0][:, :-1], self.train_fold[0][:, -1])
            y_pred = neigh.predict(self.test_fold[0][:, :-1])
            var = var + accuracy_score(self.test_fold[0][:, -1], y_pred)
            acc1 = accuracy_score(self.test_fold[0][:, -1], y_pred)
            pred1 = precision_score(self.test_fold[0][:, -1], y_pred, average='binary')
            recall1 = recall_score(self.test_fold[0][:, -1], y_pred, average='binary')

            neigh = KNeighborsClassifier(n_neighbors=i)
            neigh.fit(self.train_fold[1][:, :-1], self.train_fold[1][:, -1])
            y_pred = neigh.predict(self.test_fold[1][:, :-1])
            var = var + accuracy_score(self.test_fold[1][:, -1], y_pred)
            acc2 = accuracy_score(self.test_fold[1][:, -1], y_pred)
            pred2 = precision_score(self.test_fold[1][:, -1], y_pred, average='binary')
            recall2 = recall_score(self.test_fold[1][:, -1], y_pred, average='binary')

            neigh = KNeighborsClassifier(n_neighbors=i)
            neigh.fit(self.train_fold[2][:, :-1], self.train_fold[2][:, -1])
            y_pred = neigh.predict(self.test_fold[2][:, :-1])
            var = var + accuracy_score(self.test_fold[2][:, -1], y_pred)
            acc3 = accuracy_score(self.test_fold[2][:, -1], y_pred)
            pred3 = precision_score(self.test_fold[2][:, -1], y_pred, average='binary')
            recall3 = recall_score(self.test_fold[2][:, -1], y_pred, average='binary')
          
            neigh = KNeighborsClassifier(n_neighbors=i)
            neigh.fit(self.train_fold[3][:, :-1], self.train_fold[3][:, -1])
            y_pred = neigh.predict(self.test_fold[3][:, :-1])
            var = var + accuracy_score(self.test_fold[3][:, -1], y_pred)
            acc4 = accuracy_score(self.test_fold[3][:, -1], y_pred)
            pred4 = precision_score(self.test_fold[3][:, -1], y_pred, average='binary')
            recall4 = recall_score(self.test_fold[3][:, -1], y_pred, average='binary')
         
            neigh = KNeighborsClassifier(n_neighbors=i)
            neigh.fit(self.train_fold[4][:, :-1], self.train_fold[4][:, -1])
            y_pred = neigh.predict(self.test_fold[4][:, :-1])
            var = var + accuracy_score(self.test_fold[4][:, -1], y_pred)
            acc5 = accuracy_score(self.test_fold[4][:, -1], y_pred)
            pred5 = precision_score(self.test_fold[4][:, -1], y_pred, average='binary')
            recall5 = recall_score(self.test_fold[4][:, -1], y_pred, average='binary')
            INFO(f"k: {i}, accuracy: {acc1}, precision: {pred1}, recall:{recall1}\n accuracy: {acc2}, precision: {pred2}, recall:{recall2}\n accuracy: {acc3}, precision: {pred3}, recall:{recall3}\n accuracy: {acc4}, precision: {pred4}, recall:{recall4}\n accuracy: {acc5}, precision: {pred5}, recall:{recall5}\n")
            pdb.set_trace()


class HW32():
    def __init__(self, args) -> None:
        self.data = pd.read_csv('../data/emails.csv', sep=",")
        df = df.drop(columns=df.columns[0])
        INFO(f"data: {self.data.shape}")

        # pdb.set_trace()
 
    def run1(self):
        fold1_train = df.iloc[:1000, :]
        fold1_test = df.iloc[1000:, :]

        fold2_train = df.iloc[1000:2000, :]
        fold2_test = pd.concat([df.iloc[:1000, :], df.iloc[2000:, :]])

        fold3_train = df.iloc[2000:3000, :]
        fold3_test = fold2_test = pd.concat([df.iloc[:2000, :], df.iloc[3000:, :]])

        fold4_train = df.iloc[3000:4000, :]
        fold4_test = fold2_test = pd.concat([df.iloc[:3000, :], df.iloc[4000:, :]])

        fold5_train = df.iloc[4000:5000, :]
        fold5_test = df.iloc[:3000, :]

        model1fold = LogisticRegression(max_iter=1000)
        w, b, l = train(fold1_train.iloc[:, :-1], fold1_train.iloc[:,-1], bs=100, epochs=1000, lr=0.01)
        predicted_c = predict(fold1_test.iloc[:, :-1])
        print("Custom 1fold"+str(accuracy(np.array(fold1_test.iloc[:, -1]),predicted_c)))

        model1fold.fit(fold1_train.iloc[:, :-1], fold1_train.iloc[:,-1])
        predicted_classes = model1fold.predict(fold1_test.iloc[:, :-1])

        acc = accuracy_score(np.array(fold1_test.iloc[:, -1]),predicted_classes)
        print("1 F Precision " + str(precision_score(np.array(fold1_test.iloc[:, -1]),predicted_classes, average='binary')))
        print("1 F Recall " + str(recall_score(np.array(fold1_test.iloc[:, -1]),predicted_classes, average='binary')))
        print("SKLearn 1fold"+str(acc))

        parameters = model1fold.coef_

        model2fold = LogisticRegression(max_iter=1000)
        model2fold.fit(fold2_train.iloc[:, :-1], fold2_train.iloc[:,-1])
        predicted_classes = model2fold.predict(fold2_test.iloc[:, :-1])
        acc = accuracy_score(np.array(fold2_test.iloc[:, -1]),predicted_classes)
        print("SKLearn 2fold"+str(acc))
        print("2 F Precision " + str(precision_score(np.array(fold2_test.iloc[:, -1]),predicted_classes, average='binary')))
        print("2 F Recall " + str(recall_score(np.array(fold2_test.iloc[:, -1]),predicted_classes, average='binary')))

        w, b, l = train(fold2_train.iloc[:, :-1], fold2_train.iloc[:,-1], bs=100, epochs=1000, lr=0.01)
        predicted_c = predict(fold2_test.iloc[:, :-1])

        print(accuracy(np.array(fold2_test.iloc[:, -1]),predicted_c))


        parameters = model2fold.coef_

        model3fold = LogisticRegression(max_iter=1000)
        model3fold.fit(fold3_train.iloc[:, :-1], fold3_train.iloc[:,-1])
        predicted_classes = model3fold.predict(fold3_test.iloc[:, :-1])
        acc = accuracy_score(np.array(fold3_test.iloc[:, -1]),predicted_classes)
        print("SKLearn 3fold"+str(acc))
        print("3 F Precision " + str(precision_score(np.array(fold3_test.iloc[:, -1]),predicted_classes, average='binary')))
        print("3 F Recall " + str(recall_score(np.array(fold3_test.iloc[:, -1]),predicted_classes, average='binary')))

        w, b, l = train(fold3_train.iloc[:, :-1], fold3_train.iloc[:,-1], bs=100, epochs=1000, lr=0.01)
        predicted_c = predict(fold3_test.iloc[:, :-1])
        print("Custom 3fold"+str(accuracy(np.array(fold3_test.iloc[:, -1]),predicted_c)))

        parameters = model3fold.coef_

        model4fold = LogisticRegression(max_iter=1000)
        model4fold.fit(fold4_train.iloc[:, :-1], fold4_train.iloc[:,-1])
        predicted_classes = model4fold.predict(fold4_test.iloc[:, :-1])
        acc = accuracy_score(np.array(fold4_test.iloc[:, -1]),predicted_classes)
        print("SKLearn 4fold"+str(acc))
        print("4 F Precision " + str(precision_score(np.array(fold4_test.iloc[:, -1]),predicted_classes, average='binary')))
        print("4 F Recall " + str(recall_score(np.array(fold4_test.iloc[:, -1]),predicted_classes, average='binary')))

        w, b, l = train(fold4_train.iloc[:, :-1], fold4_train.iloc[:,-1], bs=100, epochs=1000, lr=0.01)
        predicted_c = predict(fold4_test.iloc[:, :-1])
        print("Custom 4fold"+str(accuracy(np.array(fold4_test.iloc[:, -1]),predicted_c)))

        parameters = model4fold.coef_

        model5fold = LogisticRegression(max_iter=1000)
        model5fold.fit(fold5_train.iloc[:, :-1], fold5_train.iloc[:,-1])
        predicted_classes = model5fold.predict(fold5_test.iloc[:, :-1])
        acc = accuracy_score(np.array(fold5_test.iloc[:, -1]),predicted_classes)
        print("SKLearn 5fold"+str(acc))
        print("5 F Precision " + str(precision_score(np.array(fold5_test.iloc[:, -1]),predicted_classes, average='binary')))
        print("5 F Recall " + str(recall_score(np.array(fold5_test.iloc[:, -1]),predicted_classes, average='binary')))

        w, b, l = train(fold5_train.iloc[:, :-1], fold5_train.iloc[:,-1], bs=100, epochs=1000, lr=0.01)
        predicted_c = predict(fold5_test.iloc[:, :-1])
        print("Custom 5fold"+str(accuracy(np.array(fold5_test.iloc[:, -1]),predicted_c)))

    def sigmoid(self, z):
        return 1.0/(1 + np.exp(-z))

    def loss(self, y, y_hat):
        loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
        return loss

    def gradients(self, X, y, y_hat):
        m = X.shape[0]
        dw = (1/m)*np.dot(X.T, (y_hat - y))
        db = (1/m)*np.sum((y_hat - y)) 
        return dw, db

    def normalize(self, X):
        m, n = X.shape
        for i in range(n):
            X = (X - X.mean(axis=0))/X.std(axis=0)
        return X

    def train(self, X, y, bs, epochs, lr):
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape
        w = np.zeros((n,1))
        b = 0
        y = y.reshape(m,1)
        losses = []
        for epoch in range(epochs):
            for i in range((m-1)//bs + 1):
                start_i = i*bs
                end_i = start_i + bs
                xb = X[start_i:end_i]
                yb = y[start_i:end_i]
                y_hat = sigmoid(np.dot(xb, w) + b)
                dw, db = gradients(xb, yb, y_hat)
                w -= lr*dw
                b -= lr*db
            l = loss(y, sigmoid(np.dot(X, w) + b))
            losses.append(l)
        return w, b, losses

    def predict(self, X):
        preds = sigmoid(np.dot(X, w) + b)
        pred_class = []
        pred_class = [1 if i > 0.5 else 0 for i in preds]
        
        return np.array(pred_class)

    def accuracy(self, y, y_hat):
        accuracy = np.sum(y == y_hat) / len(y)
        return accuracy

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--wxy", type=int, help='wxy')
   

    args = parser.parse_args()

    # hw31 = HW31(args)
    # hw31.run1()

    # hw32 = HW32(args)
    # hw32.run1()

    plot4()
    

   
