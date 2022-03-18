import os 
import shutil 
import cv2 
import time 

def distribute_category(dir_path, save_path, target_cat = {}):
    extentions = ['.jpg', '.JPG', '.png', '.PNG']
    dir_path_file_list = os.listdir(dir_path)
    count = 0
    print('전체 이미지 개수 : ', len(dir_path_file_list)//2)
    for file_name in dir_path_file_list:
        root, child = os.path.splitext(file_name)
        if child == '.txt':
            count +=1
            txt_source_path = os.path.join(dir_path, root + '.txt')
            for ext in extentions:
                if os.path.isfile(os.path.join(dir_path, root + ext)):
                    # 파일 확장자 명어떤건지 찾음
                    source_image_path = os.path.join(dir_path, root + ext)
                    break

            string = ''
            with open(txt_source_path, 'r') as rd :
                lines = rd.readlines()
                for box in lines:
                    box = box.strip()
                    box = box.split()
                    class_num = box[0]
                    for key, value in target_cat.items():

                        if class_num == key: 
                            # 만약 del이 target이라면 string을 생성하지 않고 continue 
                            if value == 'del':
                                break
                            else:
                                # txt의 클래스를 타겟 벨류로 변경 후 저장 
                                converted_class = value
                                string = string + "{} {} {} {} {}\n".format(converted_class, box[1], box[2], box[3], box[4])
                                break
                # 만약 값이 아무것도 없다면 확인 
                # if string == '':
                #     image = cv2.imread(source_image_path)
                #     cv2.imshow('qwe', image)
                #     cv2.waitKey(0)

            txt_save_path = os.path.join(save_path, root + '.txt')
            image_save_path = os.path.join(save_path, root + ext)
            with open(txt_save_path, 'w') as wd :
                wd.write(string)
            # shutil.move(source_image_path, image_save_path)
            shutil.copy(source_image_path, image_save_path)
            print('진행률 : {:.2f}\t {}/{} '.format(100*(count/(len(dir_path_file_list)//2)),count,len(dir_path_file_list)//2), end='\r')


# 건기연, 용인 15종 데이터 분류 
# car, bus, truck-s-a, truck-s-b, truck-m-a, truck-m-b, truck-m-c, 

# 8종 데이터 분류 = 6종에서 category 6,번을 7로 움겨야함 
# car, bus-s, bus-m, truck-s, truck-m, truck-x, motor, undefined

# 6종 데이터 분류 
# car, bus-s, bus-m, truck-s, truck-m, truck-x, undefined

category_15_to_6 = {
    '0':'0', # car -> car 
    '1':'2', # bus -> bus-m
    '2':'3', # truck-s-a -> truck-s
    '3':'3', # truck-s-b -> truck-s
    '4':'4', # truck-m-3W -> truck-m
    '5':'4', # truck-m-4W -> truck-m
    '6':'4', # truck-m-5W -> truck-m
    '7':'5', # truck-4W-ST -> truck-x
    '8':'5', # truck-4W-FT -> truck-x
    '9':'5', # truck-5W-ST -> truck-x
    '10':'5', # truck-5W-FT -> truck-x
    '11':'5', # truck-6W-ST -> truck-x
    '12':'del', #  special vehicle -> del 
    '13':'del', # motor -> del
    '14':'del', # military -> del 
    '15':'del', # undefined -> del 
}
category_15_to_8 = {
    '0':'0', # car -> car 
    '1':'2', # bus -> bus-m
    '2':'3', # truck-s-a -> truck-s
    '3':'3', # truck-s-b -> truck-s
    '4':'4', # truck-m-3W -> truck-m
    '5':'4', # truck-m-4W -> truck-m
    '6':'4', # truck-m-5W -> truck-m
    '7':'5', # truck-4W-ST -> truck-x
    '8':'5', # truck-4W-FT -> truck-x
    '9':'5', # truck-5W-ST -> truck-x
    '10':'5', # truck-5W-FT -> truck-x
    '11':'5', # truck-6W-ST -> truck-x
    '12':'del', #  특수차량 삭제 
    '13':'6', # motor -> category_8 의 motor 
    '14':'del', # military -> del 
    '15':'del', # undefined -> del 
}
category_8_to_6 = {
    '0':'0',# car -> car 
    '1':'1', # bus-s -> bus-s 
    '2':'2', # bus-m -> bus-m 
    '3':'3', # truck-s -> truck-s 
    '4':'4', # truck-m -> truck-m 
    '5':'5', # truck-x -> truck-x
    '6':'del', # motor -> delete 
    '7':'del', # undefiend -> del
}
category_6_to_8 = {
    '0':'0', # car -> car 
    '1':'1', # bus-s -> bus-s 
    '2':'2', # bus-m -> bus-m 
    '3':'3', # truck-s -> truck-s 
    '4':'4', # truck-m -> truck-m 
    '5':'5', # truck-x -> truck-x
    '6':'del', # undefined -> del
}
custom_category = {
    '4' : '6' # coco motor -> category_8 motor
}



# # 코코 -> 8종
# dir_path = 'E:/work/img_augmentation/normalized_coco'
# save_path = 'D:/데이터 정리/8종/cocomotor/'
# time1 = time.time()
# distribute_category(dir_path, save_path, custom_category)
# print("\n첫번째 걸린 시간 {:.2f}".format( 1000(time.time() - time1)))

# # 건기연 15종 -> 8종 
# dir_path = 'D:/건기연/건기연 학습용 데이터 원본/' qweqweqweqweqwe
# save_path = 'D:/data_clean/category_8'
# time1 = time.time()
# distribute_category(dir_path, save_path, category_15_to_8)
# print("\n두번째 걸린 시간 {:.2f}".format( 1000(time.time() - time1)))

# 용인 15종 -> 8종
root_dir_path = ['D:/data_clean/category_15/2022_03_08_23694_15_types', 'D:/data_clean/category_15/2022_03_17_23120_15_types']
save_path = 'D:/data_clean/category_8/all'
time1 = time.time()
for dir_path in root_dir_path:
    distribute_category(dir_path, save_path, category_15_to_8)
print("\n세번째 걸린 시간 {:.2f}".format( 1000(time.time() - time1)))


# # 용인 8종 -> 6종
# distribute_category(dir_path, save_path, custom_category)
# # 용인 15종 -> 6종
# distribute_category(dir_path, save_path, custom_category)
# # 건기연 15종 -> 6종 
# distribute_category(dir_path, save_path, custom_category)