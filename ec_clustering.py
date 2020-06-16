from utils import *

class edge_ECClustering:
    def __init__(self, **kwargs):
        if 'X' in kwargs:
            self.X = kwargs.get('X')
        if 'delta' in kwargs:
            self.delta = kwargs.get('delta')
        if 'n_clusters' in kwargs:
            self.n_clusters = kwargs.get('n_clusters')
    def edge_summarize(self):
        centers,labels = edge_summarize(self.X, self.n_clusters*100, self.delta, copy_data = False)
        self.centers = centers
        return centers
    def deliverLabel_allcenter(self, centers):
        ''' receive centers from cloud and recalculate the belonging of points in edge 
        Parameters:
        ----------------------------------
            centers - (np.array) recieved from cloud for labeling dataset in edge server
        '''
        centers = np.array([c.center_ for c in centers])
        labels = np.zeros([len(self.X),], dtype=np.int)
        mass = 0
        for i_p, p in enumerate(self.X):
            dis_list = [np.linalg.norm(center - p) for center in centers]
            cls_center = np.argmin(dis_list)
            labels[i_p] = cls_center
            mass += dis_list[cls_center]
        self.labels_ = labels
        self.mass = mass

class cloud_ECClustering:
    def __init__(self, **kwargs):
        if 'n_clusters' in kwargs:
            self.n_clusters = kwargs.get('n_clusters')
    def cloud_clustering(self, centers):
        self.centers = cloud_clustering(centers, self.n_clusters)
        return self.centers