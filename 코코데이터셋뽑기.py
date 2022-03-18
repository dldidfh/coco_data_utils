import cv2 
import os 
import shutil
import json 

def make_normalize(coco_path, save_path):
    # coco 형식의 데이터를 yolo 형식으로 변경 
    for qwe in os.listdir(coco_path):
        root, child = os.path.splitext(qwe)
        if child == '.jpg':
            join_path = os.path.join(coco_path, root)
            image = cv2.imread(join_path + '.jpg')
            h, w = image.shape[:2]
            string = ''
            with open(join_path+'.txt') as rd :
                boxes = rd.readlines()
                for box in boxes:
                    box = box.strip()
                    box = box.split()
                    xmin = float(box[1]) / w
                    ymin = float(box[2]) / h
                    xmax = (float(box[1])+float(box[3])) / w 
                    ymax = (float(box[2])+float(box[4])) / h 
                    
                    box_width = xmax - xmin 
                    box_height = ymax - ymin 

                    center_x = xmin + box_width/2
                    center_y = ymin + box_height/2

                    string = string + "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(box[0], center_x, center_y, box_width, box_height)
            # normalize 한 뒤 저장 할 위치 지정 후 저장 
            normed_save_path = os.path.join(save_path, root)
            with open(normed_save_path + '.txt', 'w') as wd:
                wd.write(string)
            shutil.copy(join_path + '.jpg', normed_save_path + '.jpg')

def unnormalize(image_path, txt_path):
    # Yolo 형식의 데이터를 화면에 박스를 그려서 윈도우에 띄움 
    for qwe in os.listdir(image_path):
        root, child = os.path.splitext(qwe)
        if child == '.jpg':
            join_path = os.path.join(image_path, root)
            image = cv2.imread(join_path + '.jpg')
            h, w = image.shape[:2]
            txt_join_path = os.path.join(txt_path, root)
            with open(txt_join_path+'.txt') as rd :
                boxes = rd.readlines()
                for box in boxes:
                    box = box.strip()
                    box = box.split()
                    center_x = float(box[1])
                    center_y = float(box[2])
                    box_width = float(box[3])
                    box_height = float(box[4])

                    xmin = int((center_x - box_width/2)*w)
                    ymin = int((center_y - box_height/2)*h)
                    xmax = int((center_x + box_width/2)*w)
                    ymax = int((center_y + box_height/2)*h)
                    cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255,0,0),2)
            cv2.imshow('qwe', image)
            cv2.waitKey(0)

def pop_target_category_in_coco(train_annot_path, target_category, image_file_path):
    # coco json 파일에서 원하는 카테고리의 데이터를 뽑아옴
    with open(train_annot_path) as t_rd:
        json_data = json.load(t_rd)
        print('전체 데이터셋은 : ',len(json_data['annotations']))
        for one_annot in json_data['annotations']:
            category_id = one_annot['category_id']
            # 만약 원하는 카테고리에 해당 아이디가 있으면 저장, 이동 
            if category_id in target_category:
                image_id = str(one_annot['image_id'])
                if image_id == '581189':
                    print('123')
                bbox = one_annot['bbox']
                # 이미지파일명은 12자이므로 12자로 변경 
                if len(image_id) < 12:
                    file_name = '0'*(12 - len(image_id)) + image_id 
                    image_path = file_name + '.jpg'
                # 해당 이미지가 이미지폴더에 존재하는지 확인 
                if os.path.isfile(os.path.join(image_file_path, image_path)):
                    # 해당 파일 목적 폴더로 이동 
                    shutil.move(os.path.join(image_file_path, image_path), os.path.join(save_path, image_path))
                txt_path = os.path.join(save_path, file_name + '.txt')
                # if os.path.isfile(txt_path):
                with open(txt_path, 'a') as awd:
                    string = "{} {} {} {} {}\n".format(category_id, bbox[0], bbox[1], bbox[2], bbox[3] )
                    awd.write(string)
"""
    카테고리 id 
    1 : person 
    2 : bicycle 
    3 : car 
    4 : motorcycle 
    6 : bus
    8 : truck

    images 에서 순서대로, annotations에서 순서대로 
    images -> file_name 
    annotations -> bbox, image_id
                    bbox -> category_id, 0 ~ 3
"""
target_category = [4]
train_annot_path = 'D:/cocodataset/annotations/instances_train2017.json'
image_file_path = 'D:/cocodataset/train2017/'

coco_path = 'coco_data/'
save_path = 'normalized_coco'            
normed_txt_patg = 'normalized_coco'

pop_target_category_in_coco(train_annot_path, target_category, image_file_path)
make_normalize(coco_path, save_path)
unnormalize(normed_txt_patg, normed_txt_patg)