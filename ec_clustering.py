from utils import *

class edge_ECClustering:
    ''' Edge functions for EC-Clustering algorithm
    Functions:
    -----------------------------------------------
        __init__ - Initiate edge object

        edge_summarize - Function for edge summarization

        deliverLabel_allcenter - Receive centers from cloud and recalculate the belonging of points in edge 
    Attributs:
    -----------------------------------------------
        X - (np.array) local dataset for edge
        delta - (float) parameter to estimate cluster number in X
        n_clusters - (int) parameter to limit maximum uploading cluster number
        centers - k-means cluster centers generated from X or sent by cloud
        mass - sum of distence from points in X to the closest centers
        labels - clustering results    
    '''
    def __init__(self, **kwargs):
        ''' Initiate edge object
        Parameters:
        -------------------------------------------
            X - (np.array) local dataset for edge
            delta - (float) parameter to estimate cluster number in X
            n_clusters - (int) parameter to limit maximum uploading cluster number
        '''
        if 'X' in kwargs:
            self.X = kwargs.get('X')
        if 'delta' in kwargs:
            self.delta = kwargs.get('delta')
        if 'n_clusters' in kwargs:
            self.n_clusters = kwargs.get('n_clusters')
    def edge_summarize(self):
        ''' Function for edge summarization
        --------------------------------------------
            Details in utils.edge_summerize
        Return:
        --------------------------------------------
            centers - (list of k_means_center) k-means cluster centers generated from X
        '''
        centers,labels = edge_summarize(self.X, self.n_clusters*100, self.delta, copy_data = False)
        self.centers = centers
        return centers
    def deliverLabel_allcenter(self, centers):
        ''' Receive centers from cloud and recalculate the belonging of points in edge 
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
    ''' Cloud functions for EC-Clustering algorithm
    Functions:
    -----------------------------------------------
        __init__ - Initiate cloud object

        cloud_clustering - Function for cloud clustering
    Attributs:
    -----------------------------------------------
        n_clusters - (int) parameter of k-means set by users
        centers - (list of edge_center )k-means cluster centers for broadcasting   
    '''
    def __init__(self, **kwargs):
        ''' Initiate edge object
        Parameters:
        -------------------------------------------
            n_clusters - (int) parameter of k-means set by users
        '''
        if 'n_clusters' in kwargs:
            self.n_clusters = kwargs.get('n_clusters')
    def cloud_clustering(self, in_centers):
        ''' Function for edge summarization
        Parameter:
        --------------------------------------------
            in_centers - (list of k_means_center objects) uploading by edges
        Return:
        --------------------------------------------
            out_centers - (list of edge_centers) broacast centers
        '''
        out_centers = cloud_clustering(centers, self.n_clusters)
        self.centers = out_centers
        return out_centers