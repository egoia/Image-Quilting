import numpy as np
from pathlib import Path
from PIL import Image


def k_means(nb_classe, X):
    indices = np.random.choice(len(X), nb_classe, replace=False)
    Y = X[indices].copy()
    dic = {i: [] for i in range(nb_classe)}
    for it in range(10000000000000):  # condition d'arret d'urgence
        for x in X:
            dist = float("Inf")
            for i in range(nb_classe):
                sample_dist = np.linalg.norm(x-Y[i])#distance euclidienne
                if sample_dist < dist :
                    dist = sample_dist
                    cluster_index = i
            dic[cluster_index].append(x)
        # barycentre
        Y_updt = np.copy(Y)
        for i in range(len(Y)):
            Y_updt[i] = np.mean(dic[i], axis=0)
        if np.linalg.norm(Y - Y_updt) < 1e-4:
            break;
        else:
            Y = Y_updt
            dic = {i: [] for i in range(nb_classe)}
    return Y

def Evaluate(x, clusters):
    dist_min = float("Inf")
    cluster = -1
    for y, index in clusters :
        dist = np.linalg.norm(y-x)
        if dist>dist_min :
            dist_min = dist
            cluster = index
    return cluster

def PrepareDataForClassification(trainingDataFolderPath, testDataFolderPath):
    return (PrepareFolderForClassification(trainingDataFolderPath), PrepareFolderForClassification(testDataFolderPath))

def PrepareFolderForClassification(folderPath):
    dataFolder = Path(folderPath)
    dataSet = []
    for img_path in dataFolder.glob("*png"):
        img = np.array(Image.open(img_path))
        hist_r = np.histogram(img[:, :, 0].ravel(), bins=256, range=(0, 256))[0]
        hist_g = np.histogram(img[:, :, 1].ravel(), bins=256, range=(0, 256))[0]
        hist_b = np.histogram(img[:, :, 2].ravel(), bins=256, range=(0, 256))[0]
        data = np.concatenate([hist_r, hist_g, hist_b])
        dataSet.append(data)
    return dataSet;