from fastsam import FastSAM, FastSAMPrompt
import os
import json
import numpy as np
import math
import torch
from tqdm import tqdm
from PIL import Image


def run_fastsam(images_folder_path, output_dir, device=torch.device('cpu'), iou=0.7):
    model_path = "./weights/FastSAM-x.pt"
    retina = True
    conf = 0.4
    imgsz = 800

    model = FastSAM(model_path)

    image_file_list = os.listdir(images_folder_path)
    everything_results = []
    for i in np.arange(0,len(image_file_list),50):
        batch = [os.path.join(images_folder_path, image_file_list[x]) for x in range(i, min(i+50, len(image_file_list)))]
        results = model(
            batch,
            device=device,
            retina_masks=retina,
            imgsz=imgsz,
            conf=conf,
            iou=iou
        )
        everything_results.extend(results)

    ann_dict = {}
    for result in tqdm(everything_results):
        prompt_process = FastSAMPrompt(result.path, [result], device=device)

        try:
            ann_every = prompt_process.everything_prompt()

            points_list, points_label = get_point_list(os.path.basename(result.path), gt_dict)
            ann = prompt_process.point_prompt(points=points_list, pointlabel=points_label)
            ann_dict[os.path.basename(result.path)] = ann

            prompt_process.plot(
                annotations=ann,
                output_path=os.path.join(output_dir, "images", os.path.basename(result.path)),
                bboxes=None,
                points=points_list,
                point_label=points_label,
                withContours=True,
                better_quality=False,
            )
            prompt_process.plot(
                annotations=ann_every,
                output_path=os.path.join(output_dir, "images_every", os.path.basename(result.path)),
                bboxes=None,
                points=points_list,
                point_label=points_label,
                withContours=True,
                better_quality=False,
            )
        except:
            continue

    return ann_dict


def get_gt_dict(gt_file_path):
    with open(gt_file_path, "r") as f:
        gt_list = json.load(f)

    gt_dict = {}
    for w in gt_list:
        gt_dict[w["filename"]] = w

    return gt_dict


def get_bounding_box(filename, gt_dict, dx=50):
    xmin = min(map(lambda x: x[0], gt_dict[filename]["junc"]))
    xmax = max(map(lambda x: x[0], gt_dict[filename]["junc"]))
    ymin = min(map(lambda x: x[1], gt_dict[filename]["junc"]))
    ymax = max(map(lambda x: x[1], gt_dict[filename]["junc"]))
    return [[max(xmin - dx, 0), max(ymin - dx, 0), min(xmax + dx, 799), min(ymax + dx, 799)]]


def find_point_on_line_at_distance(x1, y1, x2, y2, d):
    if x1 == x2:
        m = np.inf
    else:
        m = (y2 - y1) / (x2 - x1)

    if m == np.inf:
        x, y = int(x1), int(y1 + np.sign(y2 - y1) * d)
    else:
        dx = (d / math.sqrt(1 + (m * m)))
        dy = abs(m * dx)
        x = int(x1 + np.sign(x2 - x1) * dx)
        y = int(y1 + np.sign(y2 - y1) * dy)

    return x, y


def get_point_list(filename, gt_dict, d=50):
    points_list = []
    points_label = []

    # Points to be included
    for junc in gt_dict[filename]["junc"]:
        x1, y1 = junc
        x2, y2 = 400, 400
        x, y = find_point_on_line_at_distance(x1, y1, x2, y2, d)
        points_list.append([x, y])
        points_label.append(1)

    # Points to be excluded
    bbox = get_bounding_box(filename, gt_dict, dx=d)
    min_x, min_y, max_x, max_y = bbox[0]
    edges = [[min_x, min_y, min_x, max_y],
             [min_x, max_y, max_x, max_y],
             [max_x, max_y, max_x, min_y],
             [max_x, min_y, min_x, min_y]]

    for edge in edges:
        len_edge = int(math.sqrt((edge[0] - edge[2]) ** 2 + (edge[1] - edge[3]) ** 2))
        for d in np.arange(0, len_edge, 20):
            x, y = find_point_on_line_at_distance(edge[0], edge[1], edge[2], edge[3], d)
            points_list.append([int(x), int(y)])
            points_label.append(0)

    return points_list, points_label


if __name__ == "__main__":
    test_folder = "/Users/sahilmodi/Projects/Image_SuperResolution/building-outline-detection/services/roof_line_detection/cache/test_717_multidate_baseline"
    images_folder_path = os.path.join(test_folder, 'images')
    gt_file_path = [os.path.join(test_folder, gt_file) for gt_file in os.listdir(test_folder) if "gt_" in gt_file][0]
    output_dir = "output/test_717_multidate_baseline/"

    gt_dict = get_gt_dict(gt_file_path)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    ann_dict = run_fastsam(images_folder_path, output_dir, device=device, iou=0.7)
    fastsam_result = {}
    for imgname in ann_dict.keys():
        ann = ann_dict[imgname]
        mask = np.where(ann == 1)
        if not mask[1].size:
            ymin, ymax, xmin, xmax = 0, 800, 0, 800
        else:
            ymin, ymax, xmin, xmax = min(mask[1]), max(mask[1]), min(mask[2]), max(mask[2])
        fastsam_result[imgname] = [int(xmin), int(ymin), int(xmax), int(ymax)]

    with open(os.path.join(output_dir, "fastsam_result.json"), "w") as f:
        json.dump(fastsam_result, f)
