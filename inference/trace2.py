import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.build import build_model, make_loss, make_optimizer, make_scheduler
from dataset_objects.dataloader import make_data_loader
from PIL import Image
import torchvision.transforms as T
import yaml
from models.trainer import PPE_multitask_net
import torch
import cv2
from ultralytics import YOLO
# from ultralytics.yolo.utils.plotting import Annotator, colors
from collections import OrderedDict
import copy
import pandas as pd
import numpy as np
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str)
parser.add_argument("--exp_name", type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("--thr", type=float, default=0.5)
args = parser.parse_args()

gown_map = {0: 'gc', 1: 'ga', 2: 'gi'}
eyewear_map = {0: 'pr', 1: 'gg', 2: 'fc', 3: 'ea'}
mask_map = {0: 'nc', 1: 'mi', 2: 'ma'}
glove_map = {0: 'hc', 1: 'ha'}
ppe_type = ['gown', 'mask', 'eyewear', 'glove']

threshold = {i: args.thr for i in range(12)}
# with open('ablation_random_search/optimal_threshold.pkl', 'rb') as f:
#     threshold = pickle.load(f)

def main():
    video_input_path = args.video_path #"input_videos/VP3.mp4"
    output_folder = f"output_videos/{args.exp_name}"
    print("=============See here=============")
    print(output_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video_name = video_input_path.split('/')[-1][:-4]
    video_output_path = f"{output_folder}/{video_name}.mp4"
    excel_second_output_path = f"{output_folder}/{video_name}_second.xlsx"
    excel_frame_output_path = f"{output_folder}/{video_name}_frame.xlsx"
    excel_final_output_path = f"{output_folder}/{video_name}_final.xlsx"
    
    yolo_model = "data/provider_detector_yolo11m.pt"

    with open(args.model_path, 'r') as file:
        cfg = yaml.safe_load(file)

    columns1 = ['id','frame', 'gown', 'mask', 'eyewear', 'glove', 'checkpoint']
    columns2 = ['id','second', 'gown', 'mask', 'eyewear', 'glove']
    columns3 = ['id', 'gc', 'ga', 'gi', 'pr', 'gg', 'fc', 'ea', 'nc', 'mi', 'ma', 'hc', 'ha']
    df_track_PPE_id_each_frame = pd.DataFrame(columns=columns1)
    df_track_PPE_id_each_second = pd.DataFrame(columns=columns2)
    df_final_output = pd.DataFrame(columns=columns3)
    super_dict_for_display = {}


    model = build_model(cfg)
    optimizer = make_optimizer(cfg, model)
    train_loader, val_loader, test_loader = make_data_loader(cfg)
    scheduler = make_scheduler(cfg, optimizer, train_loader)
    PPE_multilabel_pl = PPE_multitask_net.load_from_checkpoint(
    model=model,
    checkpoint_path = cfg["SAVED_MODEL_PATH"], 
    loss_fn=make_loss(cfg),
    optimizer=make_optimizer(cfg, model),
    scheduler=make_scheduler(cfg, optimizer, train_loader),
    attribute_list=cfg["MODEL"]["NUMNBER_OF_CLASSES"],
    cfg=cfg
    )

    cap = cv2.VideoCapture(video_input_path)

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # video_output = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    
    provider_tracker = YOLO(yolo_model)
    results = provider_tracker.track(source=video_input_path, stream=True, device=0, verbose=True, tracker="botsort.yaml")
    for frame_num, result in enumerate(results):
        annotated_frame = result.plot()

        frame = result.orig_img
        num_id_per_img = 0

        for box in result.boxes:

            if box.id is None:
                continue

            num_id_per_img += 1
            bbox = [int(x) for x in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = bbox

            if box.cls.item() == 0:
                cropped_frame = frame[y1:y2, x1:x2]
                processed_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                processed_frame = Image.fromarray(processed_frame)

                transform = T.Compose([T.Resize([224, 224]), 
                        T.ToTensor(), 
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
                
                processed_frame = transform(processed_frame)
                processed_frame = processed_frame.unsqueeze(0)
                processed_frame = processed_frame.to(PPE_multilabel_pl.device)

                output_dict = run_attr_model(processed_frame, PPE_multilabel_pl, box.id.item(), frame_num)
                new_row = pd.DataFrame(output_dict, index=[0])
                if box.id.item() not in df_track_PPE_id_each_frame['id'].values:
                    new_row['checkpoint'] = 1

                df_track_PPE_id_each_frame = pd.concat([df_track_PPE_id_each_frame, new_row], ignore_index=True)

                prev_len = len(df_track_PPE_id_each_second.loc[df_track_PPE_id_each_second['id'] == box.id.item()])
                
                if box.id.item() not in df_track_PPE_id_each_second['id'].values: 
                    new_row = pd.DataFrame({'id': box.id.item(), 
                                            'second': frame_num//30, 
                                            'gown': output_dict['gown'], 
                                            'mask': output_dict['mask'], 
                                            'eyewear': output_dict['eyewear'], 
                                            'glove': output_dict['glove']}, index=[0])
                    #concat
                    df_track_PPE_id_each_second = pd.concat([df_track_PPE_id_each_second, new_row], ignore_index=True)

                df_track_PPE_id_each_second = consolidate_last_30_frames(df_track_PPE_id_each_frame, df_track_PPE_id_each_second, box.id.item())
                
                df_track_PPE_specific_id_each_second = df_track_PPE_id_each_second.loc[df_track_PPE_id_each_second['id'] == box.id.item()]
                is_added_new_row = len(df_track_PPE_specific_id_each_second) != prev_len
                
                # columns_to_interpolate = ['gown', 'mask', 'eyewear', 'glove']
                # dict_ppe_duration = interpolate(df_track_PPE_specific_id_each_second, columns_to_interpolate)
                
                if box.id.item() not in df_final_output['id'].values:
                    new_row = pd.DataFrame([[box.id.item()] + [0]*(len(columns3)-1)], columns=columns3)
                    df_final_output = pd.concat([df_final_output, new_row], ignore_index=True)

                df_final_output, super_dict_for_display = interpolate_na(df_final_output, df_track_PPE_specific_id_each_second, super_dict_for_display, is_added_new_row)

                for i, (key, value) in enumerate(super_dict_for_display[box.id.item()].items()):
                    text = f"{key}: {value[0]} {value[1]}sec"
                    # text = "hello"
                    text_x = x1 + 10
                    text_y = y1 + 30 + i * 30
                    cv2.putText(annotated_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        frame_width, frame_height = cv2.getTextSize(f"#Frame: {frame_num}", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(annotated_frame, f"#Frame: {frame_num}", (width - frame_width - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # video_output.write(annotated_frame)
    # video_output.release()

    # df_track_PPE_id_each_frame.to_excel(excel_frame_output_path, index=False)
    df_track_PPE_id_each_second.to_excel(excel_second_output_path, index=False)

    # df_final_output = interpolate_tail_na(df_final_output, df_track_PPE_id_each_second)
    # df_final_output.to_excel(excel_final_output_path, index=False)


def run_attr_model(input_image, PPE_multilabel_pl, id, frame):
    PPE_multilabel_pl.eval()
    output_dict = OrderedDict([
        ("gown", None),
        ("eyewear", None),
        ("mask", None),
        ("glove", None)
    ])

    with torch.no_grad():
        attr_output = PPE_multilabel_pl(input_image)
        for i, attr_output in enumerate(attr_output):
            attr_sigmoid = torch.sigmoid(attr_output)
            filtered_values = attr_sigmoid * (attr_sigmoid > threshold[i]).float()

    output_dict["gown"] = 'na' if torch.sum(filtered_values[0:3]) == 0 else gown_map[torch.argmax(filtered_values[0:3]).item()]
    output_dict["eyewear"] = 'na' if torch.sum(filtered_values[3:7]) == 0 else eyewear_map[torch.argmax(filtered_values[3:7]).item()]
    output_dict["mask"] = 'na' if torch.sum(filtered_values[7:10]) == 0 else mask_map[torch.argmax(filtered_values[7:10]).item()]
    output_dict["glove"] = 'na' if torch.sum(filtered_values[10:]) == 0 else glove_map[torch.argmax(filtered_values[10:]).item()]

    output_dict['id'], output_dict['frame'], output_dict['checkpoint'] = id, frame, 0
    return output_dict

def consolidate_last_30_frames(df, df_second, id):
    new_dict = OrderedDict([
        ("id", id),
        ("second", None),
        ("gown", None),
        ("mask", None),
        ("eyewear", None),
        ("glove", None)
    ])

    new_df = df[df['id'] == id]
    last_frame_num = new_df['frame'].iloc[-1]
    last_frame_num_where_checkpint_is_1 = new_df[new_df['checkpoint'] == 1]['frame'].iloc[-1]

    if last_frame_num  - last_frame_num_where_checkpint_is_1 >= 30:
        #update the checkpint of row of df where the frame_num is last_frame_num and id is id
        df.loc[(df['frame'] == last_frame_num) & (df['id'] == id), 'checkpoint'] = 1

        #grab the rows from the last row to the last 30 rows
        new_last_30_rows= new_df.iloc[-30:]
        for col in new_df.columns:
            if col == 'id' or col == 'frame' or col == 'checkpoint':
                continue

            new_dict[col] = new_last_30_rows[col].value_counts().index[0]

        new_dict['second'] = last_frame_num//30
        new_dict = pd.DataFrame(new_dict, index=[0])
        df_second = pd.concat([df_second, new_dict], ignore_index=True)

    curr_len_df_second_id = len(df_second.loc[df_second['id'] == id])
    return df_second


def interpolate_na(df_final_output, df_id_second, super_dict_for_display, is_added_new_row, max_na_to_interpolate=1000):
    id = df_id_second.iloc[-1]['id']    
    if is_added_new_row:
        dict_for_display = {'gown':['na', 0],
                    'mask': ['na', 0],
                    'eyewear': ['na', 0],
                    'glove': ['na', 0]}
        for ppe in dict_for_display:
            current_prediction = df_id_second.iloc[-1][ppe]
            last_non_na_prediction = 'na'
            count_na = 0
            for index in range(len(df_id_second)-1, -1, -1):
                if df_id_second.iloc[index-1][ppe] != 'na':
                    last_non_na_prediction = df_id_second.iloc[index-1][ppe]
                    break
                else:
                    count_na+=1

            #when last_non_na_pred is 'na'
            if last_non_na_prediction == 'na' and current_prediction != 'na':
                if count_na <= max_na_to_interpolate:
                    df_final_output.loc[df_final_output['id'] == id, current_prediction] += count_na+1
                else:
                    df_final_output.loc[df_final_output['id'] == id, current_prediction] += 1

            elif last_non_na_prediction != 'na' and current_prediction != 'na':
                if current_prediction == last_non_na_prediction and count_na <= max_na_to_interpolate:
                    df_final_output.loc[df_final_output['id'] == id, current_prediction] += count_na+1
                elif current_prediction != last_non_na_prediction or count_na > max_na_to_interpolate:
                    df_final_output.loc[df_final_output['id'] == id, current_prediction] += 1

            elif last_non_na_prediction == 'na' and current_prediction == 'na':
                continue

            if current_prediction == 'na':
                dict_for_display[ppe][0] = last_non_na_prediction
                dict_for_display[ppe][1] = int(df_final_output.loc[df_final_output['id'] == id, last_non_na_prediction].iloc[0])
            else:
                dict_for_display[ppe][0] = current_prediction
                dict_for_display[ppe][1] = int(df_final_output.loc[df_final_output['id'] == id, current_prediction].iloc[0])

        super_dict_for_display[id] = dict_for_display

    return df_final_output, super_dict_for_display


#the N/A tail (occluded when they leave the camera) will be interpolated
#however, during the real-time process, we don't know if the id can be still appear
#therefore, this interpolation is applied at the end of the video
def interpolate_tail_na(df_final_output, df_second):
    #group by id
    df_second_grouped_by_id = df_second.groupby('id')

    for id, df_second_id in df_second_grouped_by_id:
        for ppe in ppe_type:
            #check the last row of id of ppe column
            if df_second_id.iloc[-1][ppe] == 'na':
                #iterate from the last row of the column until the value is not 'na'
                for index in range(len(df_second_id)-1, -1, -1):
                    if df_second_id.iloc[index-1][ppe] != 'na':
                        ppe_before_disappear = df_second_id.iloc[index-1][ppe]
                        number_of_na = len(df_second_id) - index
                        df_final_output.loc[df_final_output['id'] == id, ppe_before_disappear] += number_of_na
                        break

            else:
                continue

    return df_final_output


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    total_time_seconds = end_time - start_time
    output_folder = f"output_videos/{args.exp_name}"
    video_name = args.video_path.split('/')[-1][:-4]
    log_path = os.path.join(output_folder, f"{video_name}_time.txt")
    with open(log_path, "w") as f:
        f.write(f"{total_time_seconds:.2f}\n")

    # import sys
    # print(sys.path)