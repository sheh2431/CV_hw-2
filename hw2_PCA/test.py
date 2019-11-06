import sys
import csv
import numpy as np
from numpy import *
import cv2
import glob
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

origin_h = 56
origin_w = 46
person_8_image_6 = 54
different_person = 40
images_per_person = 10
training_imgs = 7
testing_imgs = 3
training_set = different_person * training_imgs

def PCA(input_data):

    input_data = int32(input_data)
    h, w = input_data.shape                             # training_set * (h*w)          280*2576
    mean_face = np.mean(input_data, axis=0)             # (h*w)                         2576
    
    # ---- plot the mean face ---- #
    plt.imshow(mean_face.reshape(origin_h, origin_w), cmap='gray')
    #plt.show()
    cv2.imwrite(output_folder + "mean_face.png", mean_face.reshape(origin_h, origin_w))
    # ----    end of plot     ---- # 

    diff = input_data[:,:] - np.tile(mean_face, (h, 1)) # training_set * (h*w)          280*2576
    c_matrix = np.dot(diff, diff.T)                     # training_set * training_set   280*280
    eigen_value, eigen_vector = np.linalg.eig(c_matrix) # eigen_value = training_set    280, 
                                                        # eigen_vector = training_set * training_set  280*280
    eigSortIndex = argsort(-eigen_value)
    k_eigen_vector = np.zeros((training_set, k)).astype(dtype='float64')
    count = 0
    while (count < k):
        for i in range(training_set):
            if(eigSortIndex[i] == count):
                k_eigen_vector[:, count] = eigen_vector[:, i]
                count+=1
                break
    eigen_faces = np.array(np.dot(k_eigen_vector.T, diff))

    #eigen_faces = np.dot(input_data.T, k_eigen_vector)
    #eigen_faces = eigen_faces.T
    # ---- plot eigen_faces ---- #
    row = np.floor(sqrt(k))
    col = np.ceil(k/row)
    for i in range(k):
        plt.subplot(row, col, 1+i)
        #plt.tight_layout()
        plt.imshow(eigen_faces[i, :].real.reshape(origin_h, origin_w), cmap='gray')
        cv2.imwrite(output_folder + str(i+1) + "_eigenface.png", eigen_faces[i, :].real.reshape(origin_h, origin_w))
        #plt.xlabel("width")
        #plt.ylabel("height")
        #plt.title("The "+str(i+1)+"-th eigenface.")
    plt.show()
    # ----    end of plot   ---- #
    project = np.dot(k_eigen_vector, eigen_faces) + np.tile(mean_face, (h, 1))
    #w = np.array([np.dot(eigen_faces,i) for i in diff])
    #print(w.shape)
    MSE = mean_squared_error(input_data[person_8_image_6, :],project[person_8_image_6,:])

    # ---- person_8_image_6 ---- # 
    plt.subplot(1, 2, 1)
    plt.imshow(input_data[person_8_image_6, :].reshape(origin_h, origin_w), cmap='gray')
    plt.title("Original Person[8]_image[6]")
    plt.subplot(1, 2, 2)
    plt.imshow(project[person_8_image_6,:].reshape(origin_h, origin_w), cmap='gray')
    plt.title("n=" + str(k) + "\nMSE: "+str(MSE))
    plt.show()
    file_name = "reconstruct_output/"+str(k)+".png"
    cv2.imwrite(file_name, project[person_8_image_6,:].reshape(origin_h, origin_w))
    return mean_face, eigen_faces, eigen_vector, w

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
    for person in range(1, 41):
        for imgs in range(s_img, e_img+1): 
            data_img = data_folder + '/' + str(person) + "_" + str(imgs) + ".png"
            filenames.append(data_img)

    img = [Image.open(fn) for fn in filenames]
    FaceMat = np.asarray([np.array(im).flatten() for im in img])
    #asarray 在來源為ndarray結構時，會跟著來源端的array變動數值（不另佔空間）
    #flatten 可以把array或mat摺疊為1維數列
    return FaceMat


if __name__ == "__main__":
    data_folder = sys.argv[1]
    output_folder = sys.argv[2]
    k = int(sys.argv[3])
    
    FaceMat = loadImageSet(data_folder, 1, 7)
    mean, eigen_faces, eigenvector, w = PCA(FaceMat)
    #Test_FaceMat = loadImageSet(data_folder, 8, 10)
    #testing(Test_FaceMat, mean, eigen_faces, w)