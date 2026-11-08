import numpy as np

def sortFeatures(mixedFeatures, labels, opt):
        
    sortedFeatures = []
    for i in range(opt.num_classes):
        sortedFeatures.append([])
    
    for i, l in enumerate(labels):
        l = l.item()   
        feature = mixedFeatures[i]
        feature = feature.reshape([-1])
        sortedFeatures[l].append(feature)
        
    # Attention the #samples for each class are different
    return sortedFeatures


def sortData(mixedData, opt):

    sortedData = []
    for i in range(opt.num_classes):
        sortedData.append([])

    for (img, l) in mixedData:
        sortedData[l].append(img)

    return sortedData


def EuclideanDistance(feature1, feature2):
    
    return np.linalg.norm((feature1 - feature2))


def EuclideanStat(inlierFeatures):
    
    means = []
    for cFeatures in inlierFeatures:
        mean = np.mean(cFeatures, axis=0)
        means.append(mean)
        
    return means
