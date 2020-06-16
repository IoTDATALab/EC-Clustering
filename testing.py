from ec_clustering import *

EDGE_NUM = 5

# A3 iid experiment
input_path = r'.\DATA\A3_iid_%d.csv'
input_lable = r'.\DATA\A3_iid_%d_label.csv'

edge_list = []
a3_delta = 0.001
a3_clusters = 50
edge_summary = []
for e_i in range(EDGE_NUM):
    data = np.loadtxt(input_path%(e_i))
    edge = edge_ECClustering(X = data, n_clusters = a3_clusters, delta = a3_delta)
    summary = edge.edge_summarize()
    edge_list.append(edge)

    for c_i, center in enumerate(summary):
        edge_summary.append(edge_center(kcenter = center))

cloud = cloud_ECClustering(n_clusters = a3_clusters)
cloud_centers = cloud.cloud_clustering(edge_summary)
for edge in edge_list:
    edge.deliverLabel_allcenter(cloud_centers)

# f1-measure
results = []
all_labels = []
for edge in edge_list:
    results.extend(edge.labels_)
for e_i in range(EDGE_NUM):
    all_labels.extend(np.loadtxt(input_lable%(e_i),dtype=np.int))
all_labels = np.array(all_labels)
results = np.array(results)
print(f1(results, all_labels))
    