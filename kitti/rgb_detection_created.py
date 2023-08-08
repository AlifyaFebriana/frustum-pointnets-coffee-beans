import os

def yolo_to_absolute(yolo_ann, img_width, img_height):
    class_label, x_rel, y_rel, width_rel, height_rel = yolo_ann
    x_center = x_rel * img_width
    y_center = y_rel * img_height
    width = width_rel * img_width
    height = height_rel * img_height

    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    return int(class_label), int(x_min), int(y_min), int(x_max), int(y_max)

def process_file(input_file, output_file, img_dir, ann_dir):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            #label, image_id = line.strip()#.split(' ')
            filename = line.strip()
            #image_id = image_id[1:-1] 
            #filename = f"{label} ({image_id})"
            image_path = f"{img_dir}/{filename}.jpg"
            ann_path = f"{ann_dir}/{filename}.txt"

            # check the existing file
            if not os.path.exists(ann_path):
                print(f"Annotation for {filename} not found. skipping...")
                continue

            with open(ann_path, 'r') as anno_file:
                for anno_line in anno_file:
                    yolo_annotation = list(map(float, anno_line.strip().split()))
                    class_label, x_min, y_min, x_max, y_max = yolo_to_absolute(yolo_annotation, img_width=256, img_height=256)
                    outfile.write(f"{image_path} {class_label} {0.98} {x_min} {y_min} {x_max} {y_max}\n")

# process the data

train_file = '/home/junaid/alifya/Bean/frustum-pointnets-coffee-beans/kitti/image_sets/train.txt'
val_file = '/home/junaid/alifya/Bean/frustum-pointnets-coffee-beans/kitti/image_sets/val.txt'

process_file(train_file, 'kitti/rgb_detections/rgb_detection_train.txt', 'dataset/KITTI/object/training/image_2', 'kitti/labels_yolo/')
process_file(val_file, 'kitti/rgb_detections/rgb_detection_val.txt', 'dataset/KITTI/object/training/image_2', 'kitti/labels_yolo/')
