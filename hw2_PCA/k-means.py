import sys
import cv2
import os
import random
from numpy import *
from PIL import Image
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt

origin_h = 56
origin_w = 46
image_size = origin_h * origin_w
different_person = 40
images_per_person = 10
weight = np.array((0.6, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))

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
    
def Project(data, eigen_value, eigen_vector, mean_face, n):
    diff = data - np.tile(mean_face, (data.shape[0], 1))
    eigSortIndex = argsort(-eigen_value)
    k_eigen_vector = np.zeros((image_size, n)).astype(dtype='float64')
    k_eigen_vector = eigen_vector[:, eigSortIndex[:n]]
    eigen_faces = k_eigen_vector.T.real
    project_vec = np.dot(eigen_faces, diff.T).T   # Psize :k * 1
        
    return project_vec


def Kmeans(data, label, output_file):
   
    #number of clusters
    k = 10
    #number of training data
    n = data.shape[0]
    print("n: ", n)
    #number of features in the data
    f = data.shape[1]
    print("f: ", f)

    plot_data = data[:,  :2]
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    #================ initial random center =================#
    rn_std = np.random.randn(k, f) * np.tile(std, (data.shape[1], 1))
    centers = rn_std + np.tile(mean, (data.shape[1], 1))

    plot_scatter(plot_data, centers, 0)
    plt.show()
    centers_old = np.zeros(centers.shape)
    centers_new = np.copy(centers)
    clusters = np.zeros(n)

    error = np.linalg.norm(centers_new - centers_old)
    iteration = 0
    print("Err: ", error)
    while error != 0:
        iteration+=1
        #================= clustring ================#
        for i in range(n):
            min_dis = np.inf
            for j in range(k):
                #print("CENTERS: \n", centers[j] )
                #print("DATA: \n", data[i] )
                distant = Euclidean(centers_new[j], data[i])
                #print("DISTANCE: \n", distant)
                if(distant < min_dis):
                    min_dis = distant
                    clusters[i] = j

        centers_old = np.copy(centers_new)

        #================= new_center ================#
        for i in range(k):
            points = [data[j] for j in range(n) if clusters[j] == i]
            if(len(points) > 0): 
                centers_new[i] = np.mean(points, axis=0)
            else:
                centers_new[i] = 0
        error = np.linalg.norm(centers_new - centers_old)
        print("Err: ", error)

        #================= plot ====================#
        for i in range(k):
            points = np.array([data[j] for j in range(n) if clusters[j] == i])
            if(len(points)> 0 ):
                plot_scatter(points, centers_new, i)
        plt.title("iteration "+str(iteration))
        plt.savefig("iteration "+str(iteration)+".png")
        plt.show()
        
    plt.title("Final Result")
    for i in range(k):
        points = np.array([data[j] for j in range(n) if clusters[j] == i])
        if(len(points)> 0 ):
            plot_scatter(points, centers_new, i)
    plt.savefig(output_file)
    plt.show()

    plt.subplot(1, 2, 1)
    plt.title("After "+str(iteration) + " times Kmeans")
    for i in range(k):
        points = np.array([data[j] for j in range(n) if clusters[j] == i])
        if(len(points)> 0 ):
            plot_scatter(points, centers_new, i)
    plt.subplot(1, 2, 2)
    plt.title("Ground Truth")
    for i in range(k):
        points = np.array([data[j] for j in range(n) if label[j] == i+1])
        if(len(points)> 0 ):
            plot_scatter(points, centers_new, i)
    plt.savefig("Kmeans_comparison.png")
    plt.show()
    return clusters

def Euclidean(x, y):
    return np.sqrt(np.dot(np.square(x-y), weight.T))
    

def plot_scatter(plot_data, centers, index):
    colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F', '#7FFFAA', '#2F4F4F', '#FF1493']#9400D3
    plt.scatter(plot_data[:,0], plot_data[:,1], c=colors[index], s=100, alpha=0.5)
    plt.scatter(centers[:,0], centers[:,1], marker='*', c=colors[0:10], s=200)
    for i in range(plot_data.shape[0]):
        plt.annotate(index, (plot_data[i,0], plot_data[i,1]))
    
    
def loadImageSet(data_folder, s_img, e_img):

    filenames = []
    label_list = []
    for person in range(1, 11):
        for imgs in range(s_img, e_img+1): 
            data_img = data_folder + '/' + str(person) + "_" + str(imgs) + ".png"
            filenames.append(data_img)
            label_list.append(person)
    img = [Image.open(fn) for fn in filenames]
    FaceMat = np.asarray([np.array(im).flatten() for im in img])
    return FaceMat,label_list


if __name__ == "__main__":
    data_folder = sys.argv[1]
    output_file = sys.argv[2]

    train_FaceMat, train_label = loadImageSet(data_folder, 1, 7)
    
    eigen_value, eigen_vector, mean_face = PCA(train_FaceMat)
    project = Project(train_FaceMat, eigen_value, eigen_vector, mean_face, 10)
    clusters = Kmeans(project, train_label, output_file)
