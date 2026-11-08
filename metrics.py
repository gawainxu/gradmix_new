import os
import pickle
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression


"""
The file used to measure the mutual information between features and labels, I(Z;Y)
"""

def parse_option():

    parser = argparse.ArgumentParser('argument for metrics')
    
    parser.add_argument("--training_feature_path", type=str, default="/features/cifar10_resnet18_1_original_data__mixup_positive_alpha_10_beta_0.3_cutmix_no_SimCLR_1.0_1.0_0.05_trail_0_128_256_600_train")
    parser.add_argument("--testing_feature_path", type=str, default="/features/cifar10_resnet18_1_original_data__mixup_positive_alpha_10_beta_0.3_cutmix_no_SimCLR_1.0_1.0_0.05_trail_0_128_256_600_test_known")

    opt = parser.parse_args()
    opt.main_dir = os.getcwd()
    opt.training_feature_path = opt.main_dir + opt.training_feature_path
    opt.testing_feature_path = opt.main_dir + opt.testing_feature_path

    return opt


def label_probs(labels):

    """
    labels: scalar labels
    """

    vals, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)

    return probs                             #-1 * np.sum([p * np.log(p) for p in probs])


def mi_features_labels(features_train, labels_train,
                       features_test, labels_test):

    features_train = np.array(features_train)
    features_test = np.array(features_test)

    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)

    lr_model = LogisticRegression(random_state=0).fit(features_train, labels_train)
    # probs_test [#features, feature_dim]
    probs_test = lr_model.predict_proba(features_test)
    print(np.mean(probs_test, axis=0))

    label_probs_test = label_probs(labels_test)
    print(label_probs_test)

    # comute mutual information
    entropies = []
    for i, pt in enumerate(probs_test):

        entropy = np.sum(np.array([p * np.log(p / pl) for pl, p in zip(label_probs_test, pt)]))
        entropies.append(entropy)
    
    return sum(entropies) / len(entropies)
 


if __name__ == "__main__":

    opt = parse_option()

    with open(opt.training_feature_path, "rb") as f:
        features_training_head, features_training_backbone, _, labels_training = pickle.load(f) 

    with open(opt.testing_feature_path, "rb") as f:
        features_testing_head, features_testing_backbone, _, labels_testing = pickle.load(f)

    mi = mi_features_labels(features_training_head, labels_training,
                       features_testing_head, labels_testing)
    print(mi)



        
    