"""
@author: Tejas Arya (ta2763)
@author: Amritha Venkataramana (axv3602)

"""
import cv2
import os


test_path = 'D:\dataset_frames/features/testing/positive'

def create_videos_for_file_number(file_path,save_path):
    img_list = []
    file_names = os.listdir(file_path)

    for files in file_names:
        files = file_path + '/' + files

        # read image
        img = cv2.imread(files)
        # resize img to avoid insufficient memory error
        # img = cv2.resize(img, None, fx=0.65, fy=0.65)
        # get the shape of image
        height, width, layers = img.shape
        size = (width, height)
        # append the resized image to img_arr
        img_list.append(img)
    save_file_name = 'dataset_labelled/features/testing/positive/' + save_path + '.mp4'
    print(save_file_name)
    VW = cv2.VideoWriter(save_file_name, cv2.VideoWriter_fourcc(*'mp4v'), 20, size)

    for i in range(len(img_list)):
        VW.write(img_list[i])

for files in os.listdir(test_path):
    if files[6:] == '_outframes':
        # print(files)
        path = test_path + '/' + files
        save_path = files[:6]
        # print(save_path)
        create_videos_for_file_number(path,save_path)