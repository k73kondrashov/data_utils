import numpy as np
import xml.etree.ElementTree as ET


def read_yolo(path):
    with open(path) as f:
        labels = np.array([line.rstrip().split() for line in f.readlines()])
    if len(labels) == 0:
        return np.array([]), np.array([])
    classes = labels[:, 0]
    boxes = labels[:, 1:]
    return classes, boxes


def write_yolo(classes, boxes, path):
    labels = np.column_stack([np.array(classes).astype(str), np.array(boxes).astype(str)])
    with open(path, 'w') as f:
        f.write('\n'.join([''.join(line) for line in labels]))


def read_xml(xml_path):
    classes = []
    boxes = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    if root.find('size') is not  None:
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        img_size = (width, height)
    else:
        img_size = None

    for member in root.findall('object'):
        bndbox = member.find('bndbox')
        box = [
           int(bndbox.find('xmin').text),
           int(bndbox.find('ymin').text),
           int(bndbox.find('xmax').text),
           int(bndbox.find('ymax').text),
        ]
        classes.append(member.find('name').text,)
        boxes.append(box)
    return classes, boxes, img_size


# def write_xml(classes, boxes, path, img_size=None, check_coords=False):
#     all_boxes_form = ''
#
#     if
#
#     for i, box in enumerate(bboxes):
#         points.append(box)
#         xmin = int(max(box[1], 0))
#         ymin = int(max(box[2], 0))
#         xmax = int(min(box[3], image.shape[1]))
#         ymax = int(min(box[4], image.shape[0]))
#         class_id = int(box[0])
#
#         box = (xmin, ymin, xmax, ymax)
#
#         if use_xml:
#             box_form = create_xml_box(class_names[class_id], box)
#             all_boxes_form += box_form
#
# def create_xml_file(folder, path, file_name, all_boxes):
#         return f'''<?xml version="1.0" ?>
#     <annotation>
#         <folder>{folder}</folder>
#         <filename>{file_name}</filename>
#         <path>{path}</path>
#         <source>
#             <database>Unknown</database>
#         </source>
#         <segmented>0</segmented>
#         {all_boxes}
#     </annotation>'''
#
#     def create_xml_box(class_name, box):
#         xmin, ymin, xmax, ymax = box
#         box_form = f'''<object>
#         <name>{class_name}</name>
#         <pose>Unspecified</pose>
#         <truncated>0</truncated>
#         <difficult>0</difficult>
#         <bndbox>
#             <xmin>{xmin}</xmin>
#             <ymin>{ymin}</ymin>
#             <xmax>{xmax}</xmax>
#             <ymax>{ymax}</ymax>
#         </bndbox>
#     </object>'''
#         return box_form