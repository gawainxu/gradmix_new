import os
import cv2
import xml.etree.ElementTree as ET

images_folder = "/home/zhi/projects/datasets/imagenet100_train"
annotations_folder = "/home/zhi/projects/datasets/bboxes_annotations_unzip"

output_folder = "/home/zhi/projects/datasets/tmp"

for sub_folder in os.listdir(annotations_folder):
    if not os.path.isdir(os.path.join(images_folder, sub_folder)):
        continue
    else:
        print(sub_folder)
    for annotation_file in os.listdir(os.path.join(annotations_folder, sub_folder)):
        file_idx = annotation_file.split(".")[0].split("_")[-1]
        img_name = sub_folder + "_" + file_idx + ".JPEG"
        img_path = os.path.join(images_folder, sub_folder, img_name)
        if not os.path.isfile(img_path):
            continue

        img = cv2.imread(img_path)
        tree = ET.parse(os.path.join(annotations_folder, sub_folder, annotation_file))
        root = tree.getroot()
        bndbox = root.find(".//bndbox")

        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        print(xmin, ymin, xmax, ymax)
        cv2.rectangle(img,(xmin, ymax),(xmax, ymin),(0,255,0),3)
        cv2.imwrite(os.path.join(output_folder, img_name), img)

