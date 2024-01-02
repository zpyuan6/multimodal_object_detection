import os
import PIL.Image 
import yaml
import cv2
from matplotlib.pyplot import imread

#均值哈希算法
def aHash(img):
    #缩放为8*8
    img=cv2.resize(img,(30,30),interpolation=cv2.INTER_CUBIC)
    #转换为灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #s为像素和初值为0，hash_str为hash值初值为''
    s=0
    hash_str=''
    #遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s=s+gray[i,j]
    #求平均灰度
    avg=s/64
    #灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if  gray[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'            
    return hash_str

def search_similar_img(orig_folder, lack_time_path):

    lack_time_img_list = yaml.safe_load(open(lack_time_path))
    lack_time_img_hash_list = []

    for img_path in lack_time_img_hash_list:
        hash_id = img_path.split("\\")[-1].split(".")[0]
        lack_time_img_hash_list.append(hash_id)

    identified_path = {}

    for root, folders, files in os.walk(orig_folder):
        for file in files:
            if file.split(".")[-1] != "xml" and file.split(".")[-1] != "txt": 
                path = os.path.join(root, file)
                img = imread(path)
                image_hash = aHash(img)
                image_hash = str(hex(int(image_hash, 2)))
                if image_hash in lack_time_img_hash_list:
                    print(os.path.join(root, file), image_hash)
                    if image_hash in identified_path:
                        identified_path[image_hash].append(os.path.join(root, file))
                    else:
                        identified_path[image_hash] = [os.path.join(root, file)]

    with open('identified_path.yaml','w') as outfile:
        yaml.dump(identified_path, outfile)


def create_multimodal_annotation():
    dataset_folder = "F:\\pest_data\\Multitask_or_multimodality\\VOCdevkit\\VOC2007\\"

    image_path = os.path.join(dataset_folder,"JPEGImages")
    multimodal_path = os.path.join(dataset_folder,"MultimodalLabel")

    if not os.path.exists(multimodal_path):
        os.makedirs(multimodal_path)

    lack_time_imgs = []

    for root, folders, files in os.walk(image_path):
        for file in files:
            img = PIL.Image.open(os.path.join(root, file))
            img_exif = img.getexif()
            # print(img_exif)
            if 306 in img_exif:
                photo_create_time = img_exif[306]
                print(photo_create_time)
            else:
                lack_time_imgs.append(os.path.join(root, file))
                print(os.path.join(root, file))

    with open('lack_imgs.yaml','w') as outfile:
        yaml.dump(lack_time_imgs, outfile)

    return lack_time_imgs


if __name__ =="__main__":
    # lack_time_imgs = create_multimodal_annotation()
    

    orig_img_folder = "F:\\pest_data\\original_image"
    lack_time_img_path = 'lack_imgs.yaml'
    search_similar_img(orig_img_folder, lack_time_img_path)
    
    