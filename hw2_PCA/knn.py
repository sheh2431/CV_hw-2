import sys
import cv2
import os
from numpy import *
from PIL import Image
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

origin_h = 56
origin_w = 46
image_size = origin_h * origin_w
different_person = 40
images_per_person = 10


def PCA(input_data):

    input_data = int32(input_data)
    h, w = input_data.shape                             # training_set * (h*w)          280*2576
    mean_face = np.mean(input_data, axis=0)             # (h*w)                         2576

    diff = input_data[:,:] - np.tile(mean_face, (h, 1)) # training_set * (h*w)          280*2576
    c_matrix = np.dot(diff.T, diff)                     # (h*w) * (h*w)   2576*2576
    c_matrix = c_matrix/(h-1)
    eigen_value, eigen_vector = np.linalg.eig(c_matrix) # eigen_value = (h*w)    2576, 
                                                        # eigen_vector = (h*w) * (h*w)  2576*2576
    return eigen_value, eigen_vector, mean_face
    
def Reconstruct(data, eigen_value, eigen_vector, mean_face, n):
    diff = data - np.tile(mean_face, (data.shape[0], 1))
    eigSortIndex = argsort(-eigen_value)
    k_eigen_vector = np.zeros((image_size, n)).astype(dtype='float64')
    k_eigen_vector = eigen_vector[:, eigSortIndex[:n]]
    eigen_faces = k_eigen_vector.T.real
    reconstruct_face = np.zeros((data.shape[0], image_size)).astype(dtype='float64')
    project = np.dot(eigen_faces, diff.T)   # Psize :k * 1
    reconstruct_face = np.dot(project.T, eigen_faces) + mean_face # reconstruct face : (1,2576)
    return reconstruct_face

def find_max_acc(data, eigen_vale, eigen_vector, mean_face, label, n, k):
    max_score = 0
    max_para = np.zeros((2, )).astype(dtype='int8')
    for i in range(len(n)):
        reconstruct_face = Reconstruct(data, eigen_value, eigen_vector, mean_face, n[i])
        print("\nn = \t", n[i])
        print("---------------------")
        for j in range(len(k)):
            score = KNN(k[j], reconstruct_face, label)
            if(score > max_score):
                max_score = score
                max_para[0] = n[i]
                max_para[1] = k[j]
    print("=======================")
    print("max_score = ", max_score)
    print("n = ", max_para[0], " k = ", max_para[1])

    return max_para

def KNN(k, data, label):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, data, label, cv=3,scoring='accuracy')
    print("k=\t", k)
    print(scores)
    print(scores.mean())
    return scores.mean()

def test_acc(test_data, eigen_value, eigen_vector, mean_face, label, n, k):
    reconstruct_face = Reconstruct(test_data, eigen_value, eigen_vector, mean_face, n)
    scores = KNN(k, reconstruct_face, label)
    print("\nTESTING")
    print("========================")
    print(scores)

def loadImageSet(data_folder, s_img, e_img):

    filenames = []
    label_list = []
    for person in range(1, 41):
        for imgs in range(s_img, e_img+1): 
            data_img = data_folder + '/' + str(person) + "_" + str(imgs) + ".png"
            filenames.append(data_img)
            label_list.append(person)
    img = [Image.open(fn) for fn in filenames]
    FaceMat = np.asarray([np.array(im).flatten() for im in img])
    return FaceMat,label_list

if __name__ == "__main__":
    data_folder = sys.argv[1]
    n_array = [3, 10, 39]
    k_array = [1, 3, 5]

    train_FaceMat, train_label = loadImageSet(data_folder, 1, 7)
    test_FaceMat, test_label = loadImageSet(data_folder, 8, 10)
    #先用PCA找到訓練組的eigen_vector跟mean_face
    eigen_value, eigen_vector, mean = PCA(train_FaceMat)
    #再用k-nn跟訓練圖原來的label比較，以3-fold cross-validation的方式找到訓練組辨識率最好的參數
    max_para = find_max_acc(train_FaceMat, eigen_value, eigen_vector, mean, train_label, n_array, k_array)
    #再用該組參數用訓練的eigen_vector跟mean_face計算測試圖的重組情況，用k-nn跟他本來的Label比較，再以3-fold cross-validation的方式計算測試圖的辨識率
    test_acc(test_FaceMat, eigen_value, eigen_vector, mean, test_label, max_para[0], max_para[1])

        