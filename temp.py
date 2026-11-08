# To visialize the change of intra-class similarities

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

def sortFeatures(mixedFeatures, labels, num_classes):
        
    sortedFeatures = []
    for i in range(num_classes):
        sortedFeatures.append([])
    
    for i, l in enumerate(labels):
        l = l.item()                         
        feature = mixedFeatures[i]
        feature = feature.reshape([-1])
        sortedFeatures[l].append(feature)
        
    # Attention the #samples for each class are different
    return sortedFeatures


def feature_stats(features):

    features = np.squeeze(np.array(features))
    mu = np.mean(features, axis=0)
    var = np.cov(features.astype(float), rowvar=False)
    
    return (mu, var)


def distances(stats, test_features):

    diss = []
    mu, var = stats
    for features in test_features:
       
       features_normalized = features - mu
       dis = mahalanobis(features, mu, np.linalg.inv(var))
       diss.append(dis)
        
    return np.array(diss)

"""
vanille_features_path = "./features/cifar10_resnet18_temp_0.05_id_4_lr_0.001_bz_256_epoch_400_train"
new_features_path =  "./features/cifar10_resnet18_temp_0.05_id_4_positive_False_negative_False_p_1.0_no_no_train"
new_features_path1 = "./features/cifar10_resnet18_temp_0.05_id_4_positive_False_negative_False_p_1.0_no_max_similarity_train"

with open(vanille_features_path, "rb") as f:
    vanille_features, _, _, vanille_labels = pickle.load(f)

with open(new_features_path, "rb") as f:
    new_features, _, _, new_labels = pickle.load(f)

with open(new_features_path1, "rb") as f:
    new_features1, _, _, new_labels1 = pickle.load(f)


# sort features
sorted_vanille_features = sortFeatures(vanille_features, vanille_labels, num_classes=6)
sorted_new_features = sortFeatures(new_features, new_labels, num_classes=6)
sorted_new_features1 = sortFeatures(new_features1, new_labels1, num_classes=6)

sorted_vanille_features_0 = np.array(sorted_vanille_features[1])
sorted_new_features_0 = np.array(sorted_new_features[1])
sorted_new_features_0_1 = np.array(sorted_new_features1[1])

stats_vanille = feature_stats(sorted_vanille_features_0)
stats_new = feature_stats(sorted_new_features_0)
stats_new1 = feature_stats(sorted_new_features_0_1)

distance_vanille = distances(stats_vanille, sorted_vanille_features_0[:4:-1])
distance_new = distances(stats_new, sorted_new_features_0[:4:-1])
distance_new1 = distances(stats_new1, sorted_new_features_0_1[:4:-1])

max_index_vanille = np.argsort(distance_vanille)
max_index_vanille = max_index_vanille[-1000:]
max_distances_vanille = distance_vanille[max_index_vanille]
max_distances_new = distance_new[max_index_vanille]
max_distances_new1 = distance_new1[max_index_vanille]

idx = range(1000)
plt.scatter(idx, max_distances_vanille, label="vanille")
#plt.scatter(idx, max_distances_new, label="new")
plt.scatter(idx, max_distances_new1, label="new1")
plt.legend()
plt.savefig("D://projects//comprehensive_OSR//temp//distance_negative_all.pdf")
"""


vanille_features_path = "./features/cifar10_resnet18_temp_0.05_id_4_lr_0.001_bz_256_epoch_400_train"
new_features_path =  "./features/cifar10_resnet18_temp_0.05_id_4_positive_False_negative_False_p_1.0_no_no_train"
new_features_path1 = "./features/cifar10_resnet18_temp_0.05_id_4_positive_False_negative_False_p_1.0_no_max_similarity_train"

vanille_features_unknown_path = "./features/cifar10_resnet18_temp_0.05_id_4_lr_0.001_bz_256_epoch_400_test_unknown"
new_features_unknown_path =  "./features/cifar10_resnet18_temp_0.05_id_4_positive_False_negative_False_p_1.0_no_no_test_unknown"
new_features_unknown_path1 = "./features/cifar10_resnet18_temp_0.05_id_4_positive_False_negative_False_p_1.0_no_max_similarity_test_unknown"

with open(vanille_features_path, "rb") as f:
    vanille_features, _, _, vanille_labels = pickle.load(f)

with open(new_features_path, "rb") as f:
    new_features, _, _, new_labels = pickle.load(f)

with open(new_features_path1, "rb") as f:
    new_features1, _, _, new_labels1 = pickle.load(f)


with open(vanille_features_path, "rb") as f:
    vanille_unknown_features, _, _, vanille_labels = pickle.load(f)

with open(new_features_path, "rb") as f:
    new_unknown_features, _, _, new_labels = pickle.load(f)

with open(new_features_path1, "rb") as f:
    new_unknown_features1, _, _, new_labels1 = pickle.load(f)


# sort features
sorted_vanille_features = sortFeatures(vanille_features, vanille_labels, num_classes=6)
sorted_new_features = sortFeatures(new_features, new_labels, num_classes=6)
sorted_new_features1 = sortFeatures(new_features1, new_labels1, num_classes=6)

sorted_vanille_features_0 = np.array(sorted_vanille_features[4])
sorted_new_features_0 = np.array(sorted_new_features[4])
sorted_new_features_0_1 = np.array(sorted_new_features1[4])

vanille_unknown_features = np.array(vanille_unknown_features)
new_unknown_features = np.array(new_unknown_features)
new_unknown_features1 = np.array(new_unknown_features1)


stats_vanille = feature_stats(sorted_vanille_features_0)
stats_new = feature_stats(sorted_new_features_0)
stats_new1 = feature_stats(sorted_new_features_0_1)

distance_vanille = distances(stats_vanille, sorted_vanille_features_0[:4:-1])
distance_new = distances(stats_new, sorted_new_features_0[:4:-1])
distance_new1 = distances(stats_new1, sorted_new_features_0_1[:4:-1])

distance_unknown_vanille = distances(stats_vanille, vanille_unknown_features)
distance_unknown_new = distances(stats_new, new_unknown_features)
distance_unknown_new1 = distances(stats_new1, new_unknown_features1)

max_index_vanille = np.argsort(distance_vanille)[-1000:]                     # the last one is the largest
max_distances_vanille = distance_vanille[max_index_vanille]        
max_distances_new = distance_new[max_index_vanille]
max_distances_new1 = distance_new1[max_index_vanille]

index_new = np.argsort(distance_new)[-1000:]
sorted_distances_new = distance_new[index_new]

index_new1 = np.argsort(distance_new1)[-1000:]
sorted_distances_new1 = distance_new1[index_new1]


index_vanille_unknown = np.argsort(distance_unknown_vanille)[-1000:]
sorted_distances_vanille_unknown = distance_unknown_vanille[index_vanille_unknown]

index_new_unknown = np.argsort(distance_unknown_new)[-1000:]
sorted_distances_new_unknown = distance_unknown_new[index_new_unknown]

index_new_unknown1 = np.argsort(distance_unknown_new1)[-1000:]
sorted_distances_new_unknown1 = distance_unknown_new1[index_new_unknown1]


idx = range(1000)
plt.scatter(idx, sorted_distances_new1, label="new1")
plt.scatter(idx, max_distances_vanille, label="vanille")
plt.scatter(idx, sorted_distances_new_unknown1, label="new unknown1")
plt.scatter(idx, sorted_distances_vanille_unknown, label="vanille unknown")
plt.legend()
plt.savefig("./temp/sorted_distances_negative_unknown4_4.pdf")

plt.close("all")

plt.scatter(idx, sorted_distances_new, label="new1")
plt.scatter(idx, max_distances_vanille, label="vanille")
plt.scatter(idx, sorted_distances_new_unknown, label="new unknown1")
plt.scatter(idx, sorted_distances_vanille_unknown, label="vanille unknown")
plt.legend()
plt.savefig("./temp/sorted_distances_positive_unknown4_4.pdf")
