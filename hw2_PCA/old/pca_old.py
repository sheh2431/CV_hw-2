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
import seaborn as sns
from sklearn.datasets import load_digits
#digits = load_digits(return_X_y=True)
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#palette = sns.color_palette("bright", 10)

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

def PCA(input_data, output_folder, data_folder):

    input_data = int32(input_data)
    h, w = input_data.shape                             # training_set * (h*w)          280*2576
    mean_face = np.mean(input_data, axis=0)             # (h*w)                         2576
    k = 280   
    # ---- plot the mean face ---- #
    plt.imshow(mean_face.reshape(origin_h, origin_w), cmap='gray')
    # plt.show()
    cv2.imwrite(output_folder + "mean_face.png", mean_face.reshape(origin_h, origin_w))
    # ----    end of plot     ---- # 

    diff = input_data[:,:] - np.tile(mean_face, (h, 1)) # training_set * (h*w)          280*2576
    c_matrix = np.dot(diff.T, diff)                     # (h*w) * (h*w)   2576*2576
    c_matrix = c_matrix/(h-1)
    eigen_value, eigen_vector = np.linalg.eig(c_matrix) # eigen_value = (h*w)    2576, 
                                                        # eigen_vector = (h*w) * (h*w)  2576*2576
    eigSortIndex = argsort(-eigen_value)
    k_eigen_vector = np.zeros((image_size, k)).astype(dtype='float64')
    count = 0
    k_eigen_vector = eigen_vector[:, eigSortIndex[:k]]
    eigen_faces = k_eigen_vector.T
    ''' 
    # ---- save top 5 eigen vector ---- #
    for i in range(5):
        cv2.imwrite(output_folder + str(i+1) + "_eigenface.png", eigen_faces[i, :].real.reshape(origin_h, origin_w))
    # ----       end of save       ---- #
    '''
    # ---- plot eigen_faces ---- #
    row = np.floor(sqrt(k))
    col = np.ceil(k/row)
    for i in range(k):
        plt.subplot(row, col, 1+i)
        #plt.tight_layout()
        #plt.imshow(eigen_faces[i, :].real.reshape(origin_h, origin_w), cmap='gray')
        if(i<5):
            save_path = output_folder +  str(i+1) + "_eigenface.png"
            save_eigen_face = np.copy(eigen_faces[i, :].real)
            save_eigen_face = save_eigen_face.reshape(1,-1)
            #print(save_eigen_face)
            #save_eigen_face = normalize(save_eigen_face*255)
            #print(save_eigen_face)
            themax = np.max(save_eigen_face)
            themin = np.min(save_eigen_face)
            therange = themax - themin
            save_eigen_face = (save_eigen_face-themin)/therange*255
            cv2.imwrite(save_path, save_eigen_face.reshape(origin_h, origin_w))
        #plt.xlabel("width")
        #plt.ylabel("height")
        #plt.title("The "+str(i+1)+"-th eigenface.")
    #plt.show()
    # ----    end of plot   ---- #
    #project = np.dot(k_eigen_vector, eigen_faces) + np.tile(mean_face, (h, 1))
    project = np.dot(eigen_faces, diff[person_8_image_6, :].T) # Psize :k * 1 
    reconstruct_face = np.dot(project.T, eigen_faces) + mean_face # reconstruct face : (1,2576) 
    reconstruct_face = reconstruct_face.real
    MSE = mean_squared_error(input_data[person_8_image_6, :], reconstruct_face)

    # ---- person_8_image_6 ---- # 
    plt.subplot(1, 2, 1)
    plt.imshow(input_data[person_8_image_6, :].reshape(origin_h, origin_w), cmap='gray')
    plt.title("Original Person[8]_image[6]")
    plt.subplot(1, 2, 2)
    plt.imshow(reconstruct_face.reshape(origin_h, origin_w), cmap='gray')
    plt.title("n=" + str(k) + "\nMSE: "+str(MSE))
    plt.show()
    file_name = "reconstruct_output/"+str(k)+".png"
    #cv2.imwrite(file_name, )
    #cv2.imwrite(file_name, project[person_8_image_6,:].reshape(origin_h, origin_w))

    mark = ['o', '2', '^', '8', 'v', '1', 'p', 'x', 's', '3']
    color = ['b', 'c', 'g', 'k', 'm', 'r', 'y']
    # ---- for testing data ----# 
    testing_data, test_label = loadImageSet(data_folder, 8, 10) # (120, 2576)
    eigen_vector_100 = eigen_vector[:, :100] # (2576,100)
    #mean_test = np.mean(testing_data, axis=0) #(1, 2576)
    diff_test = testing_data[:,:] - np.tile(mean_face, (testing_data.shape[0], 1)) # (120,2576)
    project_test = np.dot(eigen_vector_100.T, diff_test.T) #(100, 120)
    project_test = project_test.T #(120, 100)
    project_tsne = TSNE(n_components=2).fit_transform(project_test.real, test_label)   
    for i in range(0, 40):
        a, b, c = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        for j in range(0, 3):
            color = np.array([a/255.0, b/255.0, c/255.0]).reshape(1, 3)
            s_x = project_tsne[i*3+j, 0]
            s_y = project_tsne[i*3+j, 1]
            plt.scatter(x=s_x, y=s_y, c=color)
            plt.annotate(i+1, (s_x, s_y))
    plt.show()
    return mean_face, eigen_faces, eigen_vector

def testing(Test_input, mean_face, eigen_faces, w):
    print(mean_face.shape)
    h, w = Test_input.shape
    print(h, w)
    test_diff = np.subtract(Test_input, np.tile(mean_face, (h, 1)))
    print(test_diff.shape)
    test_project = np.dot(eigen_faces, test_diff)
    distance = w - test_project
    print(distance)

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
    return FaceMat,label_list

def main():
	data_folder = sys.argv[1]
	output_folder = sys.argv[2]
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	#k = int(sys.argv[3])
	train_FaceMat, train_label = loadImageSet(data_folder, 1, 7)
    #test_FaceMat, test_label = loadImageSet(data_folder, 8, 10)
	mean, eigen_faces, eigenvector = PCA(train_FaceMat, output_folder, data_folder)

if __name__ == "__main__":
    main()
	#Test_FaceMat = loadImageSet(data_folder, 8, 10)
    #testing(Test_FaceMat, mean, eigen_faces, w)
