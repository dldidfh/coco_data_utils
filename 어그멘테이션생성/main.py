import cv2 
import numpy as np 
import os 
from config import * 
from utils import * 
from transforms_list import AlbumenatationClass
list_dir_list = os.listdir(IMAGE_DIR_PATH)
album = AlbumenatationClass(0.5)
count = 0
print("전체 파일의 개수 : {}".format(len(list_dir_list)//2))
for i, file_name in enumerate(list_dir_list):
    bboxes = []
    root, child = os.path.splitext(file_name)
    if child == '.txt':
        # 텍스트 파일이면 해당 텍스트 파일과 이름이 같은 이미지파일 찾음 
        text_path = os.path.join(IMAGE_DIR_PATH, file_name)
        for ext in IMAGE_EXT:
            if os.path.isfile(os.path.join(IMAGE_DIR_PATH, root + ext)):
                # 파일 확장자 명어떤건지 찾음
                image_path = os.path.join(IMAGE_DIR_PATH, root + ext)
                break
            else:
                continue
        image = cv2.imread(image_path)
        h,w = image.shape[:2]
        with open(text_path, 'r') as rd:
            lines = rd.readlines()
            for box in lines:
                box = box.strip()
                box = box.split()
                class_num = box[0]
                # re_center_x, re_center_y, re_box_width, re_box_height = box_value_resize(box, h, w)

                # xmin = re_center_x - re_box_width//2
                # ymin = re_center_y - re_box_height//2
                # xmax = re_center_x + re_box_width//2
                # ymax = re_center_y + re_box_height//2
                # 순서 xmin, ymin, width, height, classes(클래스는 여러개 부여 가능) 
                # bboxes.append([xmin,ymin,xmax,ymax,class_num]) # pascal_voc 포멧
                bboxes.append([float(box[1]),float(box[2]),float(box[3]),float(box[4]),class_num]) # YOLO 포멧
        # 변환(augmentation) 적용
        for aug_number in range(HOW_MANY_IMAGE_PER_AUG): # 하나의 이미지에 변환을 몇번 적용할것인지 
            trans_image, trans_bboxes = album.albumentations_transform(image.copy(), bboxes)
            # 저장 위치, 저장 파일 이름 지정 
            image_save_path = os.path.join(ANNOTATION_OUTPUT_PATH, root + "_aug" + str(aug_number) + ext)
            txt_save_path = os.path.join(ANNOTATION_OUTPUT_PATH, root + "_aug" + str(aug_number) + '.txt')
            # 이미지 저장 
            cv2.imwrite(image_save_path, trans_image)
            # 텍스트 저장 
            with open(txt_save_path, 'w') as wd:
                string = ""
                for box_len, t_box in enumerate(trans_bboxes):
                    if box_len == len(trans_bboxes):
                        string += string + str("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(t_box[-1], t_box[0], t_box[1], t_box[2], t_box[3]))
                    else:
                        string += string + str("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(t_box[-1], t_box[0], t_box[1], t_box[2], t_box[3]))
                wd.write(string)
                count +=1 
            print("현재 진행 : {:.2f}%  {}/{}".format(count/ (len(list_dir_list)//2 *HOW_MANY_IMAGE_PER_AUG) *100 , count, len(list_dir_list)//2 *HOW_MANY_IMAGE_PER_AUG ))
                


