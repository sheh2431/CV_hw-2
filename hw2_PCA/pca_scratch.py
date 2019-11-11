import sys
import csv
import random
import numpy as np
from numpy import *
import cv2
import glob
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA



origin_h = 56
origin_w = 46
person_8_image_6 = 54
different_person = 40
images_per_person = 10
training_imgs = 7
testing_imgs = 3
training_set = different_person * training_imgs
testing_set = different_person * testing_imgs
image_size = origin_h * origin_w


def testing_tsne(input_data, label, mean_face, eigen_vector, n):
    test_eigen_vector = eigen_vector[:, :n] # (2576,100)
    diff_test = input_data - np.tile(mean_face, (input_data.shape[0], 1)) # (120,2576)
    project_test = np.dot(test_eigen_vector.T, diff_test.T) #(100, 120)
    project_test = project_test.T #(120, 100)
    project_tsne = TSNE(n_components=2).fit_transform(project_test.real, label)   
    for i in range(0, 40):
        a, b, c = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        for j in range(0, 3):
            color = np.array([a/255.0, b/255.0, c/255.0]).reshape(1, 3)
            s_x = project_tsne[i*3+j, 0]
            s_y = project_tsne[i*3+j, 1]
            plt.scatter(x=s_x, y=s_y, c=color)
            plt.annotate(i+1, (s_x, s_y))
    plt.savefig("t-sne_distribution.png")
    plt.show()

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
    #asarray 在來源為ndarray結構時，會跟著來源端的array變動數值（不另佔空間）
    #flatten 可以把array或mat摺疊為1維數列
    return FaceMat, label_list


def PCA_scratch(input_data, output_folder):

    input_data = int32(input_data)
    h, w = input_data.shape                             # training_set * (h*w)          280*2576
    mean_face = np.mean(input_data, axis=0)             # (h*w)                         2576
    cv2.imwrite(output_folder + "mean_face.png", mean_face.reshape(origin_h, origin_w))

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
    # =========== output eigen faces ============ #
    for i in range(n):
        if(i<5):
            save_path = output_folder +  str(i+1) + "_eigenface.png"
            save_eigen_face = np.copy(eigen_faces[i, :])
            save_eigen_face = save_eigen_face.reshape(1,-1)
            themax = np.max(save_eigen_face)
            themin = np.min(save_eigen_face)
            therange = themax - themin
            save_eigen_face = (save_eigen_face-themin)/therange*255
            cv2.imwrite(save_path, save_eigen_face.reshape(origin_h, origin_w))
    # ============= end of output =============== #

    # ============= reconstruct person 8 image 6 ================= #
    
    reconstruct_face = np.zeros((data.shape[0], image_size)).astype(dtype='float64')
    project = np.dot(eigen_faces, diff.T)                                   # Psize :k * 1
    reconstruct_face = np.dot(project.T, eigen_faces) + mean_face           # reconstruct face : (1,2576)

    MSE = mean_squared_error(data[person_8_image_6, :], reconstruct_face[person_8_image_6, :])
    # ============================================================= #
    
    # ============= plot comparison ============ # 
    if not os.path.exists("reconstruct_output"):
        os.makedirs("reconstruct_output")
    plt.subplot(1, 2, 1)
    plt.imshow(data[person_8_image_6, :].reshape(origin_h, origin_w), cmap='gray')
    plt.title("Original Person[8]_image[6]")
    plt.subplot(1, 2, 2)
    plt.imshow(reconstruct_face[person_8_image_6, :].reshape(origin_h, origin_w), cmap='gray')
    plt.title("n=" + str(n) + "\nMSE: "+str(MSE))
    plt.savefig("reconstruct_output/"+str(n)+"_comparison.png")
    plt.show()

    file_name = "reconstruct_output/"+str(n)+".png"
    cv2.imwrite(file_name, reconstruct_face[person_8_image_6, :].reshape(origin_h, origin_w))
    # ============== end of plot =============== #

    return reconstruct_face


if __name__ == "__main__":
    data_folder = sys.argv[1]
    output_folder = sys.argv[2]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    train_FaceMat, train_label = loadImageSet(data_folder, 1, 7)
    eigen_value, eigen_vector, mean_face = PCA_scratch(train_FaceMat, output_folder)

    test_FaceMat, test_label = loadImageSet(data_folder, 8, 10)

    for n in [5, 50, 150, 280]:
        reconstruct = Reconstruct(train_FaceMat, eigen_value, eigen_vector, mean_face, n)
    
    testing_tsne(test_FaceMat, test_label, mean_face, eigen_vector, 100)

