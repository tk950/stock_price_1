# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 12:32:15 2021

@author: yuki
"""
import csv
import pickle
import statistics
import numpy as np
from collections import OrderedDict
#csvファイルを読み込んで配列に落とす
def csv2array(fName):
    with open(fName,encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        array = [row for row in reader]
        array = [[float(v) for v in row]for row in array]
    return array[:100]

def data_spliter(array):
    #最初の2つをぶち抜く。返り値は素材データと答えデータ
    material_data = [[0]*(len(array[0])-2) for i in range(len(array))]
    ans_data = [0]*len(array)
    for i in range(len(array)):
        for j in range(len(array[0])-2):
            material_data[i][j] = array[i][j+2]
            ans_data[i] = array[i][1]
    return material_data,ans_data

#標準化を行う関数　手作業で行う。返り値は順に標準化データ、標準化答え、標準偏差、平均値
def standardization(array,array_t):
    array_pstdev = [0]*len(array)
    array_average = [0]*len(array)
    for i in range(len(array)):
        array_pstdev[i] = statistics.pstdev(array[i])
        array_average[i] = statistics.mean(array[i])
        array_t[i] = round((array_t[i]-array_average[i])/(array_pstdev[i]+1e-7),8)
        for j in range(len(array[0])):
            array[i][j] = round((array[i][j]-array_average[i])/(array_pstdev[i]+1e-7),8)
    
    #最後にndarrayに全て変換しておく
    array = np.array(array)
    array_t = np.array(array_t)
    array_pstdev = np.array(array_pstdev)
    array_average = np.array(array_average)
    #今回の場合答えデータが0次元になっているので、行列化して転置しておく
    temp_array_t = np.reshape(array_t,(1,array_t.shape[0]))
    array_t = temp_array_t.T
    return array,array_t,array_pstdev,array_average

#2乗和誤差を返す 
def sum_squared_error(y, t):
    batch_size = y.shape[0]
    return 0.5 * np.sum((y-t)**2)/batch_size#返り値はただの数。配列とかではない1行1列
   

#活性化に使うシグモイド関数のクラス
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = 1/(1+np.exp(-x))
        self.out = out
        return out

    def backward(self,dout):
        dx = dout*(1.0-self.out)*self.out
        return dx

#順伝播
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad


#affineのクラスとして、重みとの掛け算
class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W)+self.b
        return out
    def backward(self,dout):
        dx = np.dot(dout,self.W.T)
        self.dW = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis = 0)
        return dx

#恒等関数と二乗和誤差をまとめたレイヤー
class EqualWithLoss:
    def __init__(self):
        self.loss = None #損失
        self.y = None #恒等関数の出力
        self.t = None #教師データ
    def forward(self,x,t):
        self.t = t
        self.y = x
        self.loss = sum_squared_error(self.y, self.t)
        return self.loss
    def backward(self,dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y-self.t)/batch_size
        return dx
    
#TwoLayerNetクラス　ここでレイヤを制御
class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size):
        pre_node_nums = np.array([input_size,hidden_size])
        weight_init_scales = np.sqrt(1.0/pre_node_nums)#sigmoidの初期値
        #重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_scales[0]*np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_scales[1]*np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)
        
        #レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['Sigmoid1'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])
        self.lastLayer = EqualWithLoss()
        
    
    #推論を行う
    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    #入力データx 教師データt
    def loss(self,x,t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)
    
    #誤差の平均パーセント
    def accuracy(self,x,t):
        y = self.predict(x)
        #y = np.argmax(y,axis=1)
        #if t.ndim != 1:t = np.argmax(t,axis=1)
        accuracy = np.sum(np.abs(y-t)/t)/float(x.shape[0])*100
        accuracy = (y[0],t[0])
        return accuracy
    
    #誤差順伝播
    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])
        
        return grads
    
    #誤差逆伝播
    def gradient(self,x,t):
        #forward
        self.loss(x,t)
        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        #設定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        
        return grads
    
    #パラメータを保存する
    def save_params(self,file_name = "params.pkl"):
        params = {}
        for key,val in self.params.items():
            params[key] = val
        with open(file_name,'wb') as f:
            pickle.dump(params,f)
            
    #パラメータを読み込む
    def load_params(self,file_name = "params.pkl"):
        with open(file_name,'rb') as f:
            params = pickle(f)
        for key, val in params.items():
            self.params[key] = val
        
        self.layers['Affine1'].W = self.params['W1']
        self.layers['Affine1'].b = self.params['b1']
        self.layers['Affine2'].W = self.params['W2']
        self.layers['Affine2'].b = self.params['b2']
    
#パラメータの更新を行うクラス
class SGD:
    def __init__(self,lr = 0.01):
        self.lr = lr
    
    def update(self,params,grads):
        for key in params.key():
            params[key] -= self.lr*grads[key]
    
class Momentum:#滑らかにぐにゃぐにゃ
    def __init__(self,lr = 0.01,momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self,params,grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key]+=self.v[key]

class AdaGrad:#lrを徐々に下げていく
    def __init__(self,lr = 0.01):
        self.lr = lr
        self.h = None
        
    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for key,val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key]*grads[key]
            params[key] -= self.lr*grads[key]/(np.sqrt(self.h[key])+1e-7)


class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
            
            

#訓練用のクラス
class Trainer:
    def __init__(self,network,x_train,t_train,x_test,t_test,epochs=20,mini_batch_size=100,
                 optimizer='SGD',optimizer_param={'lr':0.01},evaluate_sample_num_per_epoch=None,verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        
        #勾配更新
        optimizer_class_dict = {'sgd':SGD,'momentum':Momentum,'adaGrad':AdaGrad,'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size/mini_batch_size,1)
        self.max_iter = int(epochs*self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
    
    def train_step(self):
        batch_mask = np.random.choice(self.train_size,self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        grads = self.network.gradient(x_batch,t_batch)
        self.optimizer.update(self.network.params, grads)
        
        loss = self.network.loss(x_batch,t_batch)
        self.train_loss_list.append(loss)
        if self.verbose:print("train loss:"+str(loss))
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch+=1
            
            x_train_sample,t_train_sample = self.x_train,self.t_train
            x_test_sample,t_test_sample = self.x_test,self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch 
                x_train_sample,t_train_sample = self.x_train[:t],self.t_train[:t]
                x_test_sample,t_test_sample = self.x_test[:t],self.t_test[:t]
            
            train_acc = self.network.accuracy(x_train_sample,t_train_sample)
            test_acc = self.network.accuracy(x_test_sample,t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)
            
            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))
    
array,array_t = data_spliter(csv2array('data_1.csv'))
array,array_t,array_pstdev,array_average = standardization(array,array_t)
x_test,t_test = data_spliter(csv2array('data_2.csv'))
x_test,t_test,test_pstdev,test_average = standardization(x_test, t_test)
network = TwoLayerNet(99, 50, 1)
trainer = Trainer(network,array,array_t,x_test,t_test,epochs=10,mini_batch_size=100,optimizer='Adam',optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()
