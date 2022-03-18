
def box_value_resize(box, h , w):
    re_center_x =  float(box[1]) * w 
    re_center_y = float(box[2]) * h 
    re_box_width = float(box[3]) * w 
    re_box_height = float(box[4]) * h 

    return int(re_center_x), int(re_center_y), int(re_box_width), int(re_box_height)