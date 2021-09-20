import os
import glob
import numpy as np
from PIL import Image, ImageOps

def multi_label(imgArray, size, nb_class):
    one_hot_label = np.zeros((size, size, nb_class))
    for y in range(size):
        for x in range(size):
            if imgArray[y][x] < nb_class:
                one_hot_label[y, x, imgArray[y][x]] = 1
            #one_hot_label[y, x, 0 if imgArray[y][x] >= nb_class else imgArray[y][x]] = 1
            #one_hot_label[y, x, imgArray[y][x]] = 1
    return one_hot_label

def one_hot_label2rgb(image_data):
    (h, w, _) = image_data.shape
    rgb_img = np.zeros((h, w, 3))
    for y in range(h):
        for x in range(w):
           #!RGB becuase PIL is used:
           label = np.argmax(image_data[y][x])
           if label == 1:
               rgb_img[y][x] = [255, 0, 0]
           elif label == 2:
               rgb_img[y][x] = [0, 255, 0]
           elif label == 3:
               rgb_img[y][x] = [255, 255, 0]
           elif label == 4:
               rgb_img[y][x] = [0, 0, 255]
    return rgb_img

def load_images_train(dir_name, size, end_num, train_list, test_list, ORIGINAL_COLOR, ROTATION):
    files = []
    for i in train_list:
        #print(glob.glob(os.path.join(dir_name, '* '+str(i).zfill(5)+'_*.*')))
        if ROTATION:
            files.extend(glob.glob(os.path.join(dir_name, '* '+str(i).zfill(5)+'_*.*')))
        else:
            files.extend(glob.glob(os.path.join(dir_name, '* '+str(i).zfill(5)+'_1.*')))
    #print('load_train_images:', files)
    
    files.sort()
    im_0 = np.zeros((len(files), size, size, 3), 'float32')   
    for i, file in enumerate(files):
        srcImg = Image.open(file)
        distImg = srcImg.convert('RGB')
        imgArray = np.asarray(distImg)
        imgArray = np.reshape(
            imgArray, (size, size, 3))
        im_0[i] = imgArray / 255
    return (files, im_0)

def load_images_test(dir_name, size, end_num, test_list, ORIGINAL_COLOR, ROTATION, whole_prediction = False):
    files = []
    if whole_prediction:
        files = glob.glob(os.path.join(dir_name+'/*.png'))
        files.sort()
        files = np.array(files)[test_list]
        #for test_num in test_list:
            #print(os.path.join(dir_name, '*_' + str(test_num).zfill(1) + '.***'))
            #files.extend(glob.glob(os.path.join(dir_name, '* ' + str(test_num).zfill(5) + '.*')))
    else:
        for test_num in test_list:
            print(glob.glob(os.path.join(dir_name, '* ' + str(test_num).zfill(5) + '_*.*')))
            if ROTATION:
                files.extend(glob.glob(os.path.join(dir_name, '* ' + str(test_num).zfill(5) + '_*.*')))
            else:
                files.extend(glob.glob(os.path.join(dir_name, '* ' + str(test_num).zfill(5) + '_1.*')))
        #print('test_num', test_num, os.path.join(dir_name, '*_' + str(test_num).zfill(1) + '_*.*'))
        #files.extend(glob.glob(os.path.join(dir_name, '*_' + str(test_num).zfill(1) + '.***')))
        #files.extend(glob.glob(os.path.join(dir_name, '*_' + str(test_num).zfill(1) + '_*.*')))
        files.sort()
    print('sorted_files', files)
    images = np.zeros((len(files), size, size, 3), 'float32')
    for i, file in enumerate(files):
        srcImg = Image.open(file)
#       srcImg = srcImg.resize((128, 128), Image.LANCZOS)  
        distImg = srcImg.convert('RGB')
        imgArray = np.asarray(distImg)
        imgArray = np.reshape(
            imgArray, (size, size, 3))
        images[i] = imgArray / 255
    return (files, images)

def load_images_train_l(dir_name, size, end_num, train_list, test_list, ORIGINAL_COLOR, ROTATION, nb_class = 5):
    #nb_class = 5    

    files = []
    for i in train_list:
        #print(glob.glob(os.path.join(dir_name, '* '+str(i).zfill(5)+'_*.*')))
        if ROTATION:
            files.extend(glob.glob(os.path.join(dir_name, '* '+str(i).zfill(5)+'_*.*')))
        else:
            files.extend(glob.glob(os.path.join(dir_name, '* '+str(i).zfill(5)+'_1.*')))
    #print('load training labels', files)

    files.sort()
    im_0 = np.zeros((len(files), size, size, nb_class), 'float32')

    for i, file in enumerate(files):
        print(i, file)
        srcImg = Image.open(file)
        distImg = srcImg.convert('L')
        imgArray = np.asarray(distImg)
        one_hot_label = multi_label(imgArray, size, nb_class)
        #print(one_hot_label)
        im_0[i] = one_hot_label
    return (files, im_0)


def load_images_test_l(dir_name, size, end_num, test_list):
    files = []
    for test_num in test_list:
        files.extend(glob.glob(os.path.join(dir_name, '*_' + str(test_num).zfill(1) + '.***')))
    files.sort()
    images = np.zeros((len(files), size, size, 1), 'float32')
    for i, file in enumerate(files):
        srcImg = Image.open(file)
      
        distImg = srcImg.convert('L')
        imgArray = np.asarray(distImg)
        imgArray = np.reshape(
            imgArray, (size, size, 1))
        images[i] = imgArray / 255
    return (files, images)

def save_images(dir_name, image_data_list, file_name_list):
    for _, (image_data, file_name) in enumerate(zip(image_data_list, file_name_list)):
        name = os.path.basename(file_name)
        print(file_name)        
        rgb_img = one_hot_label2rgb(image_data)
        #print(rgb_img)
        rgb_img = rgb_img.astype(np.uint8)
        pil_img = Image.fromarray(rgb_img)
        save_path = os.path.join(dir_name, name)
        pil_img.save(save_path, 'png')
        print(save_path)

        """
        (w, h, _) = image_data.shape
        image_data = np.reshape(image_data, (w, h))
        image_data = np.uint8((image_data * 255))
        distImg = Image.fromarray(image_data)
        distImg = distImg.convert('L')
        save_path = os.path.join(dir_name, name)
        distImg.save(save_path, "png")
        print(save_path)
        """
