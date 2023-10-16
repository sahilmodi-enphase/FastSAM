import json
import os

if __name__ == '__main__':

    test_folder = "/Users/sahilmodi/Projects/Image_SuperResolution/building-outline-detection/services/roof_line_detection/cache/test_717_multidate_baseline"
    images_folder_path = os.path.join(test_folder, 'images')
    gt_file_path = [os.path.join(test_folder, gt_file) for gt_file in os.listdir(test_folder) if "gt_" in gt_file][0]
    fastsam_result_file = "/Users/sahilmodi/Projects/FastSAM/output/test_717_multidate_baseline_2/fastsam_result.json"
    image_file_list = "/Users/sahilmodi/Projects/Image_SuperResolution/building-outline-detection/services/roof_line_detection/cache/test_717_multidate_baseline/results_20231017_171036_fastsam_filtered/image_file_list.txt"

    with open(gt_file_path, "r") as f:
        gt_list = json.load(f)
    with open(fastsam_result_file, "r") as f:
        fs_dict = json.load(f)
    with open(image_file_list, "r") as f:
        image_file_list_data = f.read()
        image_file_list = image_file_list_data.split("\n")

    # Ground truth data
    gt_dict = {}
    for w in gt_list:
        gt_dict[w["filename"]] = w

    gt_list_new = []

    for filename in image_file_list:
        xmin_fs, ymin_fs, xmax_fs, ymax_fs = fs_dict[filename]
        xmin_gt = min(map(lambda x: x[0], gt_dict[filename]["junc"]))
        xmax_gt = max(map(lambda x: x[0], gt_dict[filename]["junc"]))
        ymin_gt = min(map(lambda x: x[1], gt_dict[filename]["junc"]))
        ymax_gt = max(map(lambda x: x[1], gt_dict[filename]["junc"]))

        xcenter_fs, ycenter_fs = (xmax_fs + xmin_fs) / 2, (ymax_fs + ymin_fs) / 2
        xcenter_gt, ycenter_gt = (xmax_gt + xmin_gt) / 2, (ymax_gt + ymin_gt) / 2

        x_offset = xcenter_fs - xcenter_gt
        y_offset = ycenter_fs - ycenter_gt

        x_stretch = (xmax_fs - xmin_fs) / (xmax_gt - xmin_gt)
        y_stretch = (ymax_fs - ymin_fs) / (ymax_gt - ymin_gt)

        gt_dict_new = {}
        gt_dict_new['height'], gt_dict_new['width'] = 800, 800
        gt_dict_new['filename'] = filename
        gt_dict_new['junc'], gt_dict_new['lines'] = [], []

        for junc in gt_dict[filename]['junc']:
            x, y = junc
            x_new = round(xcenter_gt + x_stretch * (x - xcenter_gt) + x_offset)
            y_new = round(ycenter_gt + y_stretch * (y - ycenter_gt) + y_offset)
            gt_dict_new['junc'].append([x_new, y_new])

        for line in gt_dict[filename]['lines']:
            x1, y1, x2, y2 = line
            x1_new = round(xcenter_gt + x_stretch * (x1 - xcenter_gt) + x_offset)
            y1_new = round(ycenter_gt + y_stretch * (y1 - ycenter_gt) + y_offset)
            x2_new = round(xcenter_gt + x_stretch * (x2 - xcenter_gt) + x_offset)
            y2_new = round(ycenter_gt + y_stretch * (y2 - ycenter_gt) + y_offset)
            gt_dict_new['lines'].append([x1_new, y1_new, x2_new, y2_new])

        gt_list_new.append(gt_dict_new)

    with open("gt_multidate_fastsam_v1.json", "w") as f:
        json.dump(gt_list_new, f)