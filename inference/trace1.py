import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO
import pandas as pd
import argparse
import os
import cv2
from ultralytics.utils.plotting import Annotator, colors
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str)
parser.add_argument("--exp_name", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--thr", type=float, default=0.5)
parser.add_argument("--si_thr", type=float, default=0.9)
args = parser.parse_args()

print("=======args======")
print("model: ", args.model)
print("video_path: ", args.video_path)
print("exp_name: ", args.exp_name)
print("thr: ", args.thr)
print("spatial_interaction: ", args.si_thr)

columns = ['id', 'frame', 'gc', 'ga', 'gi', 'pr', 'gg', 'fc', 'ea', 'nc', 'mi', 'ma', 'hc', 'ha']
other_keys = ['gc', 'ga', 'gi', 'pr', 'gg', 'fc', 'ea', 'nc', 'mi', 'ma', 'hc', 'ha']
#0, 3, 4, 5, 9, 13
cls_id_to_label = { 0: 'gc', 1:'ga', 2:'gi', 3:'pr', 4:'gg', 5:'fc', 6:'ea', 7:'nc', 8:'mi', 9:'ma', 10:'hc', 11:'ha'}
spatial_interaction = args.si_thr


def main():

    ppe_detector = YOLO(f'data/{args.model}/weights/best.pt')
    provider_tracker = YOLO("data/provider_detector_yolo11m.pt")

    video_input_path = args.video_path
    output_folder = f"output_videos/{args.exp_name}"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_name = video_input_path.split('/')[-1][:-4]
    video_output_path = f"{output_folder}/{video_name}.mp4"
    excel_second_output_path = f"{output_folder}/{video_name}_second.xlsx"
    excel_frame_output_path = f"{output_folder}/{video_name}_frame.xlsx"
    excel_final_output_path = f"{output_folder}/{video_name}_final.xlsx"

    # df_track_PPE_id_each_frame = pd.DataFrame(columns=columns)
    df_frame = pd.DataFrame(columns=['id', 'frame', 'gown', 'eyewear', 'mask', 'glove'])
    df_second = pd.DataFrame(columns=['id', 'second', 'gown', 'eyewear', 'mask', 'glove'])

    query_dict = {
                "id":[],
                "person_bbox":[],
                "gc": [],
                "ga": [],
                "gi": [],
                'pr': [],
                'gg': [],
                'fc': [],
                'ea': [],
                'nc': [],
                'mi': [],
                'ma': [],
                'hc': [],
                'ha': [],
                "frame":[]
                }

    cap = cv2.VideoCapture(video_input_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_output = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    results = provider_tracker.track(source=video_input_path, stream=True, device=0, verbose=True, tracker="botsort.yaml")
    
    for frame_num, result in enumerate(results):
        annotated_frame = result.plot()

        frame = result.orig_img
        num_id_per_img = 0

        for box in result.boxes:

            if box.id is None:
                continue

            num_id_per_img+=1
            bbox = [int(x) for x in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = bbox


            query_dict['id'].append(int(box.id.item()))
            query_dict['frame'].append(frame_num+1)
            query_dict['person_bbox'].append([x1, y1, x2, y2])

            for key in other_keys:
                query_dict[key].append(None)

        detect_ppe(ppe_detector, annotated_frame, frame, frame_num, query_dict, num_id_per_img)
        #query_dict['id'][i] = 1, query_dict['frame'][i] = 1, query_dict['ga'][i] = [conf1, conf2] or None
        #iterate ids in the current frame

        #num_id_per_img = 3, -1, -2 ,-3
        for i in range(-1, -(num_id_per_img+1), -1):
            query_row = {'id': query_dict['id'][i], 'frame': query_dict['frame'][i], 'gown': None, 'eyewear': None, 'mask': None, 'glove': None}
            query_row = determine_ppe_by_attribute(query_dict, query_row, 'gown', i, start_idx=0, end_idx=3)
            query_row = determine_ppe_by_attribute(query_dict, query_row, 'eyewear', i, start_idx=3, end_idx=7)
            query_row = determine_ppe_by_attribute(query_dict, query_row, 'mask', i, start_idx=7, end_idx=10)
            query_row = determine_glove(query_dict, query_row, 'glove', i, start_idx=10, end_idx=12)
            df_frame = pd.concat([df_frame, pd.DataFrame([query_row])], ignore_index=True)

    #     video_output.write(annotated_frame)
    # video_output.release()

    print("query_dict id length", len(query_dict['id']))
    print("query_dict frame length", len(query_dict['frame']))

    df_frame.to_excel(excel_frame_output_path, index=False)

    df_track_id_frame = df_frame.groupby('id')
    for id, subdf_id in df_track_id_frame:
        new_subdf_id_second = consolidate_30frames(subdf_id)
        df_second = pd.concat([df_second, new_subdf_id_second], ignore_index=True)

    df_second.to_excel(excel_second_output_path, index=False)
    return


def determine_ppe_by_attribute(query_dict, query_row, attr, i, start_idx, end_idx):
    attr_dict = {}
    for key in other_keys[start_idx:end_idx]:
        if query_dict[key][i] is None:
            attr_dict[key] = 0
        else:
            attr_dict[key] = max(query_dict[key][i])
    
    if all(value == 0 for value in attr_dict.values()):
        query_row[attr] = 'na'
    else:
        query_row[attr] = max(attr_dict, key=attr_dict.get)

    return query_row

def determine_glove(query_dict, query_row, attr, i, start_idx, end_idx):
    attr_dict = {}
    for key in other_keys[start_idx:end_idx]:
        if query_dict[key][i] is None:
            attr_dict[key] = [0]
        else:
            if len(query_dict[key][i]) >= 2:
                attr_dict[key] = sorted(query_dict[key][i], reverse=True)[:2]
            else:
                attr_dict[key] = query_dict[key][i]

    top_two = get_top_two_elements(attr_dict) # top_two = [(key, value), (key, value)]
    top_two = [(key, value) for key, value in top_two if value != 0]
    # print(top_two)

    if len(top_two) == 0:
        query_row[attr] = 'na'
    elif len(top_two) == 1:
        query_row[attr] = top_two[0][0]
    elif len(top_two) == 2:
        #if two keys are the same, query_row[attr] = top_two[0][0]
        if top_two[0][0] == top_two[1][0]:
            query_row[attr] = top_two[0][0]
        else:
            query_row[attr] = 'ha'

    return query_row

def get_top_two_elements(data):
    combined = [(key, value) for key, values in data.items() for value in values]
    combined.sort(reverse=True, key=lambda x: x[1])
    return combined[:2]



def check_box_overlap(person_box, pred_box, threshold):
    x1_p, y1_p, x2_p, y2_p = person_box
    x1, y1, x2, y2 = pred_box
    x_overlap = max(0, min(x2, x2_p) - max(x1, x1_p))
    y_overlap = max(0, min(y2, y2_p) - max(y1, y1_p))
    overlap_area = x_overlap * y_overlap
    box_area = (x2 - x1) * (y2 - y1)

    if overlap_area >= threshold * box_area:
        return True
    else:
        return False


def detect_ppe(detector, frame, coppied_frame, frame_num, query_dict, num_id_per_img):

    results = detector.predict(coppied_frame, device=1, conf=args.thr, verbose=False, iou=0.7) #three categories: mask, eyewear, and gloves (one pair)

    annotator = Annotator(frame, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc')

    boxes = results[0].boxes
    names = results[0].names

    # if boxes is not None:
    #     for d in reversed(boxes):
    #         cls, conf = d.cls.squeeze(), d.conf.squeeze()
    #         c = int(cls)
    #         label = (f'{names[c]}' if names else f'{c}') + (f'{conf:.2f}')
    #         annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

    for box in results[0].boxes:
        pred_box = [int(x) for x in box.xyxy[0].tolist()]

        frame_indexes = [i for i, x in enumerate(query_dict['frame']) if x == frame_num+1]
        for i in frame_indexes:
            person_box = query_dict['person_bbox'][i]

            # If the overlap area is greater than or equal to 90% of the bounding box area
            if check_box_overlap(person_box, pred_box, threshold=spatial_interaction):
                # gown detector model and head detector model are different
                # set ppe_cls from None to 1
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                text = cls_id_to_label.get(cls_id)

                if text is not None and query_dict[text][i] is None:
                    query_dict[text][i] = [conf]

                #when two gloves are inside in person box
                elif query_dict[text][i] is not None:
                    query_dict[text][i].append(conf)

                # detections_df = detections_df.append({'id': query_dict['id'][i], 'box': pred_box, 'text': text}, ignore_index=True)

def post_process(data):
    df_interpolated = pd.DataFrame(columns=['id', 'second', 'gown', 'eyewear', 'mask', 'glove'])
    df_second = pd.DataFrame(columns=['id', 'second', 'gown', 'eyewear', 'mask', 'glove'])
    df_frame = pd.DataFrame(columns=['id', 'frame', 'gown', 'eyewear', 'mask', 'glove'])
    df_final_output = pd.DataFrame(columns=['id']+other_keys)
    df_track_id_frame = data.groupby('id')

    for id, subdf_id in df_track_id_frame:
        new_subdf_id_frame = convert_view_by_adherence(subdf_id)
        df_frame = pd.concat([df_frame, new_subdf_id_frame], ignore_index=True)
        new_subdf_id_second = consolidate_30frames(new_subdf_id_frame)
        df_second = pd.concat([df_second, new_subdf_id_second], ignore_index=True)
        df_interpolated_id_second = interpolate_na(new_subdf_id_second)
        df_interpolated = pd.concat([df_interpolated, df_interpolated_id_second], ignore_index=True)
        
        entry = {'id': id}
        for key in other_keys:
            entry[key] = (df_interpolated_id_second == key).sum().sum()

        df_final_output = pd.concat([df_final_output, pd.DataFrame([entry])], ignore_index=True)

    return df_final_output, df_second


def convert_view_by_adherence(data):
    columns = ['id','frame', 'gown', 'eyewear', 'mask', 'glove']
    new_df = pd.DataFrame(columns=columns)
    #iterate by row
    for index, row in data.iterrows():
        entry = {'id': row['id'], 'frame': row['frame'], 'gown': 'na', 'eyewear': 'na', 'mask': 'na', 'glove': 'na'}
        #check gown
        if row['gc'] == 1 and row['ga'] == 0 and row['gi'] == 0:
            entry['gown'] = 'gc'
        elif row['gc'] == 0 and row['ga'] == 1 and row['gi'] == 0:
            entry['gown'] = 'ga'
        elif row['gc'] == 0 and row['ga'] == 0 and row['gi'] == 1:
            entry['gown'] = 'gi'

        #check eyewear
        if row['pr'] == 1 and row['gg'] == 0 and row['fc'] == 0 and row['ea'] == 0:
            entry['eyewear'] = 'pr'
        elif row['pr'] == 0 and row['gg'] == 1 and row['fc'] == 0 and row['ea'] == 0:
            entry['eyewear'] = 'gg'
        elif row['pr'] == 0 and row['gg'] == 0 and row['fc'] == 1 and  row['ea'] == 0:
            entry['eyewear'] = 'fc'
        elif row['pr'] == 0 and row['gg'] == 0 and row['fc'] == 0 and row['ea'] == 1:
            entry['eyewear'] = 'ea'

        #check mask
        if row['nc'] == 1 and row['mi'] == 0 and row['ma'] == 0:
            entry['mask'] = 'nc'
        elif row['nc'] == 0 and row['mi'] == 1 and row['ma'] == 0:
            entry['mask'] = 'mi'
        elif row['nc'] == 0 and row['mi'] == 0 and row['ma'] == 1:
            entry['mask'] = 'ma'

        #check glove
        if row['hc'] == 2 and row['ha'] == 0:
            entry['glove'] = 'hc'
        elif row['hc'] == 0 and row['ha'] == 1:
            entry['glove'] = 'ha'
        elif row['hc'] == 0 and row['ha'] == 2:
            entry['glove'] = 'ha'
        elif row['hc'] == 1 and row['ha'] == 1:
            entry['glove'] = 'ha'

        new_df = pd.concat([new_df, pd.DataFrame([entry])], ignore_index=True)

    return new_df


def consolidate_30frames(data):
    columns = ['id','second', 'gown', 'eyewear', 'mask', 'glove']
    new_df = pd.DataFrame(columns=columns)

    for i in range(0, len(data), 30):
        subdf = data.iloc[i:i+30]
        last_frame_num = subdf['frame'].iloc[-1]
        entry = {'id': data['id'].iloc[-1], 'second': last_frame_num//30, 'gown': 'na', 'eyewear': 'na', 'mask': 'na', 'glove': 'na'}
        for col in data.columns:
            if col == 'id' or col == 'frame':
                continue

            entry[col] = subdf[col].value_counts().index[0]

        
        new_df = pd.concat([new_df, pd.DataFrame([entry])], ignore_index=True)
    
    return new_df


def interpolate_na(df, max_na_to_interpolate=1000):
    columns = ['gown', 'mask', 'eyewear', 'glove']
    for column in columns:
        col_values = df[column].values
        
        # Step 1: Replace initial 'na' values with the first non-'na' value
        first_non_na_index = np.where(col_values != 'na')[0]
        
        if len(first_non_na_index) > 0:
            first_non_na_value = col_values[first_non_na_index[0]]
            initial_na_indices = np.where(col_values == 'na')[0]
            
            if len(initial_na_indices) > 0 and initial_na_indices[0] < first_non_na_index[0]:
                col_values[:first_non_na_index[0]] = first_non_na_value
                
    for column in columns:
        col_values = df[column].values
        na_indices = np.where(col_values == 'na')[0]
        
        if len(na_indices) == 0:
            continue
        
        # Find continuous sequences of 'na' values
        gap_start = None
        for i in range(len(na_indices)):
            if gap_start is None:
                gap_start = na_indices[i]
            if i == len(na_indices) - 1 or na_indices[i + 1] != na_indices[i] + 1:
                gap_end = na_indices[i]
                gap_length = gap_end - gap_start + 1
                
                # Interpolate if the gap length is within the allowed range
                if gap_length <= max_na_to_interpolate:
                    if gap_start > 0 and gap_end < len(col_values) - 1:
                        # Interpolate with the surrounding values
                        if col_values[gap_start - 1] == col_values[gap_end + 1]:
                            col_values[gap_start:gap_end + 1] = col_values[gap_start - 1]
                
                gap_start = None
        
        df[column] = col_values


    for column in columns:
        col_values = df[column].values
        
        if df[column].iloc[-1] == 'na' and len(np.unique(col_values)) > 1:
            for i in range(len(df)-1, -1, -1):
                if df[column].iloc[i] != 'na':
                    row_index_of_last_non_na = i
                    last_ppe = df[column].iloc[i]
                    break
            
            for i in range(row_index_of_last_non_na, len(df)):
                if df[column].iloc[i] == 'na':
                    df.loc[df.index[i], column] = last_ppe

            else:
                continue
    
    return df


if __name__ == '__main__':
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