import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

if __name__ == "__main__":
    voc_folder = "F:\\pest_data\\Multitask_or_multimodality\\YOLO_01JAN"
    annotation_folder = os.path.join(voc_folder, "labels")
    image_folder = os.path.join(voc_folder, "images")

    target_index = 18

    for root, folders, files in os.walk(annotation_folder):
        for file in files:
            image_id = file.split(".")[0]
            bboxs = np.loadtxt(os.path.join(root, file))
            print(bboxs)
            if bboxs.shape == (0,):
                continue
            if len(bboxs.shape) == 1:
                bboxs = np.expand_dims(bboxs, 0)
            class_indexs = bboxs[:, 0]

            class_indexs = class_indexs.tolist()
            if target_index in class_indexs:
                img = mpimg.imread(os.path.join(root.replace("labels", "images"), f"{image_id}.JPG"))
                plt.imshow(img)
                plt.show()




