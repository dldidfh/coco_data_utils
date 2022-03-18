from re import L
import albumentations as alb 
class AlbumenatationClass():
    def __init__(self, random_seed) -> None:
        self.transforms = alb.Compose([
            # 크기변환, 화면 전환 
            alb.OneOf([
                alb.RandomCrop(9*60,16*60,p=1),
                alb.RandomCrop(9*50,16*50,p=1),
                alb.RandomCrop(9*40,16*40,p=1),
            ],p=0.2),
            # 스케일 변화 후 회전 
            alb.ShiftScaleRotate(p=0.3, border_mode=1),
            # 수평 반전 
            alb.HorizontalFlip(p=0.8),
                                        # # 광학 왜곡 - 바운딩박스 형태에 적용 안됨 
                                        # alb.OpticalDistortion(p=1),
            # 블러 
            alb.OneOf([
                # 모션 블러 
                alb.MotionBlur(p=1),
                #                 # JPG 압축 
                #                 alb.JpegCompression(quality_lower=30, quality_upper=50, always_apply=True),
                # 미디언 블러 
                alb.augmentations.transforms.MedianBlur(p=1) ,
                # 고급 블러 advanced blur 
                alb.augmentations.transforms.AdvancedBlur(p=1),
                # 엠보싱 
                alb.augmentations.transforms.Emboss(p=1),
                # 이퀄라이즈 
                alb.augmentations.transforms.Equalize(p=1),
                # 가우시안 블러 
                alb.augmentations.transforms.GaussianBlur(p=1),
                # 샤프 
                alb.augmentations.transforms.Sharpen(p=1),
                # 노출 - 캔버스처럼 보임 
                alb.augmentations.transforms.Solarize(p=1),

            ],p=1),
            # channel 변환 
            alb.OneOf([
                # 랜덤 밝기 변화 
                alb.RandomBrightness(p=1),
                # 렌덤 감마 
                alb.RandomGamma(p=1),
                # CLAHE - 이미지 균일화 
                alb.augmentations.transforms.CLAHE(p=1) ,
                # RGB 변화 
                alb.RGBShift(p=1),
                # 색상 변환 - jittering
                alb.ColorJitter(p=1),
                                # # 채널 섞기 - 왜 오류나는지는 모르겠지만 오류남 
                                # alb.augmentations.transforms.ChannelShuffle(p=1),
                # 어두운 색 세피아 - 오징어를 뜻함 
                alb.augmentations.transforms.ToSepia(p=1),
            ],p=1),

            # # 노이즈 
            alb.OneOf([
                # 가우시안 노이즈 
                alb.augmentations.transforms.GaussNoise(p=0.8),
                # 비내리는 날씨 
                alb.augmentations.transforms.RandomRain (p=0.2, rain_type='heavy'),
                                # # 슈퍼픽셀 - 너무 노이즈가 강함 
                                # alb.augmentations.transforms.Superpixels(p=1),

            ],p=1),
            

    ], 
    bbox_params=alb.BboxParams(format='yolo', min_visibility=0.3, min_area=225))
    # bbox_params=alb.BboxParams(format='pascal_voc', min_visibility=0.3, min_area=500))
    # bbox_params=alb.BboxParams(format='yolo', min_visibility=0.3, min_area=500, label_fields=['class_labels']))

    def albumentations_transform(self, image, bboxes):
        result = self.transforms(image=image,bboxes=bboxes)
        return result['image'], result['bboxes']