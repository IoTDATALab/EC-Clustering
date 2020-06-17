# EC-Clustering
Code and datasets for article ''EC-Clustering: A Federated Clustering Algorithm For Edge-Cloud Collaborative Intelligence''
## Dependencies
- Python (>= 3.6)
- NumPy (>= 1.13.3)
- SciPy (>= 0.19.1)
- Sklearn (>= 0.11)
## Datasets in DATA
Dataset | Filename  
-|-|
IID A3 dataset | A3_iid_$edge_number$.csv |
non-IID A3 dataset | A3_noniid_$edge_number$.csv |
IID ImagenetS dataset | ImagenetS_iid_$edge_number$.csv |
non-IID ImagenetS dataset | ImagenetS_noniid_$edge_number$.csv |

2D dataset [A3](http://cs.uef.fi/sipu/datasets/a3.txt) —— a synthetic two-dimensional dataset with 50 circular clusters and 150 points for each cluster.

High-dimension dataset [ImagenetS](http://image-net.org/challenges/LSVRC/2012/) —— 4,096-dimension features extracted from images of top ten classes in ILSVRC2012-validation by VGG16(fs_conv5_2 layer).
## Example of EC-Clustering 
Here, we give an example for running EC-Clustering on A3 IID datasets.

Firstly, initialize the parameters for clustering task.

```
input_path = r'.\DATA\A3_iid_%d.csv'
input_lable = r'.\DATA\A3_iid_%d_label.csv'
a3_delta = 0.001
a3_clusters = 50
```
Then, load local dataset for initiating each edge
```
data = np.loadtxt(input_path%(e_i))
edge = edge_ECClustering(X = data, n_clusters = a3_clusters, delta = a3_delta)
```
Execute edge summarization in each edge, and receive the summary of local dataset.
```
summary = edge.edge_summarize()
```
Upload summary of edges to the cloud and execute cloud clustering.
```
cloud = cloud_ECClustering(n_clusters = a3_clusters)
cloud_centers = cloud.cloud_clustering(edge_summary)
```
Broadcast cloud clustering results to each edge.
```
for edge in edge_list:
    edge.deliverLabel_allcenter(cloud_centers)
```
We also give a f1-score measurement function `f1()` to envalue the clustering results.


