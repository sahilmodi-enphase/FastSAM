import json
import os
import re
import math
import pandas as pd

if __name__ == '__main__':

    test_folder = "/Users/sahilmodi/Projects/Image_SuperResolution/building-outline-detection/services/roof_line_detection/cache/test_717_multidate_baseline"
    images_folder_path = os.path.join(test_folder, 'images')
    gt_file_path = [os.path.join(test_folder, gt_file) for gt_file in os.listdir(test_folder) if "gt_" in gt_file][0]
    facet_info_file_path = [os.path.join(test_folder, facet_file) for facet_file in os.listdir(test_folder) if "facet_line_info" in facet_file][0]
    result_dir = os.path.join(test_folder, 'results_20231011_210152')
    wireframe_file = os.path.join(result_dir, 'wireframe.json')
    f1score_file = os.path.join(result_dir, 'hawp_image_result_analysis_0.85_10.0.csv')
    hawp_image_dir = os.path.join(result_dir, 'hawp_images')
    gt_image_dir = os.path.join(result_dir, 'ground_truth_images')
    fastsam_result_file = "/Users/sahilmodi/Projects/FastSAM/output/test_717_multidate_baseline_2/fastsam_result.json"

    with open(gt_file_path, "r") as f:
        gt_list = json.load(f)
    with open(wireframe_file, "r") as f:
        wireframe_data = json.load(f)
    with open(fastsam_result_file, "r") as f:
        fs_dict = json.load(f)

    # Hawp data
    wireframe_dict = {}
    for wd in wireframe_data:
        wireframe_dict[wd["filename"]] = wd

    #Ground truth data
    gt_dict = {}
    for w in gt_list:
        gt_dict[w["filename"]] = w

    # Hawp and Ground Truth:  F1 score
    f1score_df = pd.read_csv(f1score_file, skipfooter=20)

    f1score_df["Date"] = f1score_df.apply(lambda s: pd.to_datetime(re.findall("[0-9-]+", s["Filename"])[5]), axis=1)
    f1score_df["HAWP_Lines"] = f1score_df.apply(lambda s: sum(i > 0.85 for i in wireframe_dict[s["Filename"]]["edges-weights"]), axis=1)
    f1score_df.set_index("Filename", inplace=True)

    df = pd.DataFrame()
    for filename in fs_dict.keys():
        xmin_fs, ymin_fs, xmax_fs, ymax_fs = fs_dict[filename]
        xmin_gt = min(map(lambda x: x[0], gt_dict[filename]["junc"]))
        xmax_gt = max(map(lambda x: x[0], gt_dict[filename]["junc"]))
        ymin_gt = min(map(lambda x: x[1], gt_dict[filename]["junc"]))
        ymax_gt = max(map(lambda x: x[1], gt_dict[filename]["junc"]))
        try:
            df = df._append({"Filename": filename, "xmin_fs": xmin_fs, "xmin_gt": xmin_gt, "dxmin": xmin_fs - xmin_gt,
                             "ymin_fs": ymin_fs, "ymin_gt": ymin_gt, "dymin": ymin_fs - ymin_gt,
                             "xmax_fs": xmax_fs, "xmax_gt": xmax_gt, "dxmax": xmax_fs - xmax_gt,
                             "ymax_fs": ymax_fs, "ymax_gt": ymax_gt, "dymax": ymax_fs - ymax_gt,
                             "Lines": f1score_df.loc[filename, "Lines"],
                             "HAWP_Lines": f1score_df.loc[filename, "HAWP_Lines"],
                             "Date": f1score_df.loc[filename, "Date"],
                             "F1score": f1score_df.loc[filename, "F1score"]
                             }, ignore_index=True)
        except:
            continue

    df["dtotal"] = abs(df["dxmin"]) + abs(df["dymin"]) + abs(df["dxmax"]) + abs(df["dymax"])
    df["dtotal2"] = (df["dxmin"]**2 + df["dymin"]**2 + df["dxmax"]**2 + df["dymax"]**2).apply(math.sqrt)
