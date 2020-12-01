import numpy as np
import matplotlib.pyplot as plt

class fedcavg_edge:
    def __init__(self, **kwargs):
        if 'X' in kwargs:
            self.X = kwargs.get('X')
            self.Np = len(self.X)
        if 'K' in kwargs:
            self.K = kwargs.get('K')
            self.H = np.zeros([self.Np, self.K])
        if 'Q1' in kwargs:
            self.Q1 = kwargs.get('Q1')
        if 'Q2' in kwargs:
            self.Q2 = kwargs.get('Q2')
    def samples4W(self):
        indexs = list(range(self.Np))
        np.random.shuffle(indexs)
        return self.X[indexs[0:self.K]]
    def optimal_H_W(self, W_from_cloud):
        self.c_W = np.array(W_from_cloud)
        self.c_step = 100
        if np.sum(self.H) == 0:
            self.e_W = self.c_W
            for x_i, x in enumerate(self.X):
                distance = self.c_W - x
                distance = np.linalg.norm(distance, axis = 1)
                index = np.argmin(distance)
                self.H[x_i, index] = 1

        self.H =self._update_H(self.X, self.c_W, self.H, self.c_step, self.Q1)
        self.d_step = 1
        self.e_W = self._update_W(self.X, self.c_W, self.H, self.d_step, self.Q2)
        return self.e_W

    def _calculate_step_c(self, W_from_cloud):
        temp_matrix = np.dot(W_from_cloud, W_from_cloud.T)
        eigen,temp=np.linalg.eig(temp_matrix)
        return max(eigen.real)/2
    def _calculate_step_d(self, H):
        temp_matrix = np.dot(H.T, H)
        eigen,temp=np.linalg.eig(temp_matrix)
        return max(eigen.real)/2
    def _update_H(self, X, W, H, c_step, iters):
        new_H = None
        for iter in range(iters):
            new_H = H - 2*(np.dot(np.dot(H,W),W.T)-np.dot(X, W.T))/c_step
            
            H = new_H
        indexs = np.argmin(new_H,axis = 1)
        new_H = np.zeros(new_H.shape)
        for i in range(len(X)):
            new_H[i,indexs[i]] = 1
        return new_H
    def _update_W(self, X, W, H, d_step ,iters):
        new_W = None  
        for iter in range(iters):
            new_W = W - 2*(np.dot(np.dot(H.T, H), W)-np.dot(H.T, X))/d_step
            W = new_W
        return new_W
    def delivery_W(self, W):
        self.label = np.zeros([len(self.X),])
        for x_i, x in enumerate(self.X):
            distance = W - x
            distance = np.linalg.norm(distance, axis = 1)
            index = np.argmin(distance)
            self.label[x_i] = index
class fedcavg_cloud:
    def __init__(self, **kwargs):
        if 'K' in kwargs:
            self.K = kwargs.get('K')
    def initiate_W(self, upload_samples):
        ''' Initiate W in cloud by sampling uploaded points from edge
        '''
        upload_samples = np.array(upload_samples)
        indexs = list(range(len(upload_samples)))
        np.random.shuffle(indexs)
        self.W = upload_samples[indexs[0:self.K]]
        return self.W
    def update_W(self, upload_W_tuple):
        new_W = np.zeros(self.W.shape)
        self.N = 0
        for w, Np in upload_W_tuple:
            self.N +=Np
            new_W += w*Np
        new_W/=self.N
        self.W = new_W
        return self.W

from ec_dual_clustering import *
if __name__ == '__main__':
    # a3 k-means++
    output_path=r'D:\worktemp\k-means-exp\outputdata\imagenet500_uniform_%d.csv'
    output_label_path=r'D:\worktemp\k-means-exp\outputdata\imagenet500_uniform_%d_label.csv'

    edge_list = []
    cloud = fedcavg_cloud(K = 10)
    samples = []
    for i in range(5):
        data = np.loadtxt(output_path%(i))
        edge = fedcavg_edge(X = data, K = 10, Q1 = 10, Q2 = 1)
        samples.extend(edge.samples4W())
        edge_list.append(edge)
    cloud_W = cloud.initiate_W(samples)
    for i in range(10):
        upload = []
        for i in range(5):
            edge_W = edge_list[i].optimal_H_W(cloud_W)
            upload.append((edge_W, edge_list[i].Np))
        cloud_W = cloud.update_W(upload)
    data = []
    results = []
    labels = []
    for i in range(5):
        edge_list[i].delivery_W(cloud_W)
        data.extend(edge_list[i].X)
        results.extend(edge_list[i].label)
        labels.extend(np.loadtxt(output_label_path%(i)))
    data = np.array(data)
    print(f1(results, labels))
    

  
    
    