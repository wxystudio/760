import argparse
import logging
from logging import debug as DEBUG
from logging import info as INFO
from logging import error as ERROR
logging.basicConfig(format=f"[%(levelname)s %(filename)s line:%(lineno)d] %(message)s", level=logging.INFO)

import random
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torchvision
import os
import glob
import math

class language():
    def __init__(self):
        self.path = "languageID"
        self.languages = ['e', 'j', 's']
    def read_text_file(self, file_path):
        with open(file_path, 'r') as f:
            contents = f.read()
        return contents
    def run(self):
        self.global_count_dict = {'e':{}, 'j':{}, 's': {}}

        for language in self.languages:
            dict = {}
            for file in glob.glob(f"{path}/{language}[0-9].txt"):
                if file.endswith(".txt"):
                    file_path = f"{file}"
                    contents = self.read_text_file(file_path)
                for char in contents:
                    if char == "\n":
                        continue
                    else:
                        if dict.get(char) == None:
                            dict[char] = 1
                        else:
                            dict[char] = dict[char] + 1   
            self.global_count_dict[language] = dict

        global_ccp = {'e':{}, 'j':{}, 's': {}}
        for language in self.languages:
            ccp = {}
            total = 0
            for char in sorted(self.global_count_dict[language].keys()):
                total = total + self.global_count_dict[language][char]
            for char in sorted(self.global_count_dict[language].keys()):
                if ccp.get(char) == None:
                    ccp[char] =  float(self.global_count_dict[language][char] + 0.5)/ (total + (27 * 0.5))
            global_ccp[language] = ccp

        test_file = f"{self.path}/e10.txt"

    def predict(self, test_file):
        x_vector = {}
        for char in self.read_text_file(test_file):
            if char == "\n":
                continue
            else:
                if x_vector.get(char) == None:
                    x_vector[char] = 1
                else:
                    x_vector[char] = x_vector[char] + 1   

        log_likelihood =  {'e': float(0), 'j': float(0), 's': float(0)}
        for language in self.languages:
            ccp = self.global_ccp[language]
            logsum = 0
            for char in x_vector:
                if ccp.get(char) == None:
                    ccp[char] = 0.5 / 27*0.5
                logsum = logsum + math.log(ccp[char]) * x_vector[char]
            log_likelihood[language] = logsum

        prior = float((10 + 0.5)) / (30 + 3*0.5)
        posterior = [log_likelihood[i] * prior for i in log_likelihood]
        prediction = posterior.index(max(posterior))
        return self.languages[prediction]


class Project4():
    def __init__(self,):
        self.train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('/files/', train=True, download=True,
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                        ])),
            batch_size=64, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('/files/', train=False, download=True,
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                        ])),
            batch_size=64, shuffle=True)

    def run(self):
        (x_train, y_train), (x_test, y_test) = Transform.mnist.load_data()
        examples = enumerate(test_loader)
        batch_idx, (example_data, example_targets) = next(examples)
        network = Net()
        optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
        x_train = (x_train.reshape(len(x_train),784)/255).astype('float32')
        x_test = (x_test.reshape(len(x_test),784)/255).astype('float32')

        clf = naive_dnn(lr=0.01,ep=30,storage=[784,300,10],weight=(0,0.01))
        train_acc,test_acc = clf.train(x_train,y_train,x_test,y_test)

    def run_torch(self):
        self.network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
            INFO('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), '/results/model.pth')
            torch.save(optimizer.state_dict(), '/results/optimizer.pth')

    def test(self):
        def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        INFO('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def plot(self):
        # plot testing accuracy
        plt.plot(train_acc,color='b',label='Train Acc')
        plt.plot(test_acc,color='r',label='Test Acc')
    
    def adjust_weight(self):
        # Build a simple 2-layer feed forward network as described
        model = nn.Sequential(nn.Linear(input_size, hidden_size, bias=False),
                            nn.Sigmoid(),
                            nn.Linear(hidden_size, output_size, bias=False),
                            nn.LogSoftmax(dim=1))
        INFO(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
        # Using the cross entropy (or NLL) loss
        criterion = nn.NLLLoss()

        epochs = 20
        losses = []
        for i in range(epochs):
            running_loss = 0
            for images, labels in train_loader:
                images = images.view(images.shape[0], -1)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            else:
                losses.append(float(running_loss/len(train_loader)))
                INFO("Epoch {0}, Training loss: {1}".format(i, running_loss/len(train_loader)))

        correct_count, all_count = 0, 0
        for images,labels in val_loader:
        for i in range(len(labels)):
            img = images[i].view(1, 28*28)
            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
            correct_count += 1
            all_count += 1

        INFO("Number Of Images Tested =", all_count)
        INFO("\nModel Accuracy =", (correct_count/all_count))
    
       

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Linear(1, 10, kernel_size=5)
        self.conv2 = nn.Linear(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
class naive_dnn:
    def __init__(self, storage, ep, lr,batch ,weight):
        self.storage = storage
        self.ep = ep
        self.lr = lr
        self.batch = batch
        
        mu = weight[0]
        sig = weight[1]
        self.weights = {
            'w1':np.random.normal(mu,sig,size=(storage[1],storage[0])),
            'w2':np.random.normal(mu,sig,size=(storage[2],storage[1])),
            'w3':np.random.normal(mu,sig,size=(storage[3],storage[3]))
        }
       
    def sigmoid(self, val):
        sig = 1/(1+np.exp(-val))
        result = sig*(1-sig)
        return result
     
    
 
    def softmax(self, val):
        exp = np.exp(val - val.max())
        sig = exp/np.sum(exp,axis=0)
        result = sig*(1-sig)

    def forward(self, x):
        weights = self.weights
        weights['a0'] = x
        
        weights['z1'] = np.dot(weights['w1'],weights['a0'])
        weights['a1'] = self.sigmoid(weights['z1'])
        
        weights['z2'] = np.dot(weights['w2'],weights['a1'])
        weights['a2'] = self.softmax(weights['z2'])

        weights['z3'] = np.dot(weights['w3'],weights['a2'])
        weights['a3'] = self.softmax(weights['z3'])
        
        return weights['a3']
    

    def backward(self, y_train, y_pred):
        weights = self.weights
        delta_w = {} 
        
        err = (y_pred-y_train)/y_pred.shape[0]*self.softmax(weights['z3'],der=True)
        delta_w['w3'] = np.outer(err,weights['a2'])
        
        err = np.dot(weights['w2'].T,err)*self.sigmoid(weights['z2'],der=True)
        delta_w['w2'] = np.outer(err,weights['a1'])

        err = np.dot(weights['w2'].T,err)*self.sigmoid(weights['z1'],der=True)
        delta_w['w1'] = np.outer(err,weights['a0'])
        
        return delta_w
    
   
    def update(self,deltas):
        for key,val in deltas.items():
            self.weights[key] -= self.lr*val
    
    def accuracy(self, x_test, y_test):
        correct = np.zeros(len(y_test))
        for i,(x,y) in enumelr(zip(x_test,y_test)):
            out = self.forward(x)
            pred = np.argmax(out)
            correct[i] = 1*(pred==np.argmax(y))
        return np.mean(correct)
    
    def train(self,x_train,y_train,x_test,y_test):
        test_acc = []
        train_acc = []
        for it in range(self.ep):
            shuffle = np.random.permutation(len(y_train)) 
            for x,y in zip(x_train[shuffle],y_train[shuffle]):
                out = self.forward(x) 
                deltas = self.backward(y,out) 
                self.update(deltas)
            temp_test_acc = self.accuracy(x_test,y_test)
            temp_train_acc = self.accuracy(x_train,y_train)
            test_acc.append(temp_test_acc)
            train_acc.append(temp_train_acc)
        return train_acc,test_acc
    
    


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--wxy", type=int, help='wxy')
    
    lg = language()
    # lg.run()

    pt = Project4()
    # pt.run()
    # pt.run_torch()
    # pt.plot()
    pt.adjust_weight()