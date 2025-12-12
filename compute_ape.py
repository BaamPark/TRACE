import pandas as pd
import numpy as np
import os

ppe_cls = ['gc', 'ga', 'gi', 'pr', 'gg', 'fc', 'fi', 'sg', 'ea', 'nc', 'mi', 'ma', 'hc', 'ha']
# ppe_cls = ['gn', 'en', 'mn', 'ha']
attr = ['gown', 'mask', 'eyewear', 'glove']
videos = [
    '10.29 scenario 1 trauma 1 above bed',
    '10.29 scenario 2 trauma 1 above bed',
    '10.29 scenario 3 trauma 1 above bed',
    '10.29 scenario 4 trauma 1 above bed',
    '10.29 scenario 5 trauma 1 above bed',
#     'Vpen1',
#     'Vpen2',
#     'Vpen3'
]

def main():
    input_exp_dir = "output_videos/1125_2025_swin26_thr0.05"
    output_exp_dir = "output_videos/1125_2025_swin26_thr0.05/interpolate_headOff_bodyOff_tailOff"
    if not os.path.exists(output_exp_dir):
        os.mkdir(output_exp_dir)

    df_ground_truth = pd.DataFrame(columns=['video'] + ppe_cls)
    # df_ground_truth = pd.concat([df_ground_truth, pd.DataFrame([{'video': '10.29 scenario 1 trauma 1 above bed', 'gc': 436, 'ga': 433, 'gi': 436, 'pr': 218,'gg': 218, 'fc': 218, 'fi': 215, 'sg': 218, 'ea': 218, 'nc': 215, 'mi': 436, 'ma': 436, 'hc': 651, 'ha': 654}])], ignore_index=True)
    # df_ground_truth = pd.concat([df_ground_truth, pd.DataFrame([{'video': '10.29 scenario 2 trauma 1 above bed', 'gc': 529, 'ga': 541, 'gi': 518, 'pr': 259, 'gg': 282, 'fc': 259, 'fi': 270, 'sg': 259, 'ea': 259, 'nc': 518, 'mi': 270, 'ma': 541, 'hc': 518, 'ha': 1070}])], ignore_index=True)
    # df_ground_truth = pd.concat([df_ground_truth, pd.DataFrame([{'video': '10.29 scenario 3 trauma 1 above bed', 'gc': 304, 'ga': 304, 'gi': 297, 'pr': 152, 'gg': 152, 'fc': 152, 'fi': 152, 'sg': 145, 'ea': 152, 'nc': 304, 'mi': 161, 'ma': 288, 'hc': 449, 'ha': 456}])], ignore_index=True)
    # df_ground_truth = pd.concat([df_ground_truth, pd.DataFrame([{'video': '10.29 scenario 4 trauma 1 above bed', 'gc': 1013, 'ga': 0, 'gi': 684, 'pr': 342, 'gg': 342, 'fc': 329, 'fi': 0, 'sg': 342, 'ea': 342, 'nc': 342, 'mi': 671, 'ma': 342, 'hc': 1026, 'ha': 671}])], ignore_index=True)
    # df_ground_truth = pd.concat([df_ground_truth, pd.DataFrame([{'video': '10.29 scenario 5 trauma 1 above bed', 'gc': 372, 'ga': 186, 'gi': 372, 'pr': 186, 'gg': 186, 'fc': 186, 'fi': 0, 'sg': 186, 'ea': 186, 'nc': 186, 'mi': 372, 'ma': 186, 'hc': 558, 'ha': 372}])], ignore_index=True)
    #sg is ea
    df_ground_truth = pd.concat([df_ground_truth, pd.DataFrame([{'video': '10.29 scenario 1 trauma 1 above bed', 'gc': 436, 'ga': 433, 'gi': 436, 'pr': 218,'gg': 218, 'fc': 433, 'ea': 436, 'nc': 215, 'mi': 436, 'ma': 436, 'hc': 651, 'ha': 654}])], ignore_index=True)
    df_ground_truth = pd.concat([df_ground_truth, pd.DataFrame([{'video': '10.29 scenario 2 trauma 1 above bed', 'gc': 529, 'ga': 541, 'gi': 518, 'pr': 259, 'gg': 282, 'fc': 529, 'ea': 518, 'nc': 518, 'mi': 270, 'ma': 541, 'hc': 518, 'ha': 1070}])], ignore_index=True)
    df_ground_truth = pd.concat([df_ground_truth, pd.DataFrame([{'video': '10.29 scenario 3 trauma 1 above bed', 'gc': 304, 'ga': 304, 'gi': 297, 'pr': 152, 'gg': 152, 'fc': 304, 'ea': 297, 'nc': 304, 'mi': 161, 'ma': 288, 'hc': 449, 'ha': 456}])], ignore_index=True)
    df_ground_truth = pd.concat([df_ground_truth, pd.DataFrame([{'video': '10.29 scenario 4 trauma 1 above bed', 'gc': 1013, 'ga': 0, 'gi': 684, 'pr': 342, 'gg': 342, 'fc': 329, 'ea': 684, 'nc': 342, 'mi': 671, 'ma': 342, 'hc': 1026, 'ha': 671}])], ignore_index=True)
    df_ground_truth = pd.concat([df_ground_truth, pd.DataFrame([{'video': '10.29 scenario 5 trauma 1 above bed', 'gc': 372, 'ga': 186, 'gi': 372, 'pr': 186, 'gg': 186, 'fc': 186, 'ea': 372, 'nc': 186, 'mi': 372, 'ma': 186, 'hc': 558, 'ha': 372}])], ignore_index=True)

    # df_ground_truth = pd.concat([df_ground_truth, pd.DataFrame([{'video': 'Vpen1', 'gc': 372, 'ga': 786, 'gi': 291, 'pr': 0,'gg': 0, 'fc': 251, 'ea': 1198, 'nc': 248, 'mi': 195, 'ma': 1006, 'hc': 333, 'ha': 1116}])], ignore_index=True)
    # df_ground_truth = pd.concat([df_ground_truth, pd.DataFrame([{'video': 'Vpen2', 'gc': 350, 'ga': 698, 'gi': 182, 'pr': 0, 'gg': 0, 'fc': 521, 'ea': 709, 'nc': 355, 'mi': 480, 'ma': 395, 'hc': 563, 'ha': 667}])], ignore_index=True)
    # df_ground_truth = pd.concat([df_ground_truth, pd.DataFrame([{'video': 'Vpen3', 'gc': 186, 'ga': 880, 'gi': 143, 'pr': 0, 'gg': 0, 'fc': 428, 'ea': 781, 'nc': 254, 'mi': 493, 'ma': 462, 'hc': 413, 'ha': 796}])], ignore_index=True)

    df_pred_individual = pd.DataFrame(columns=['video'] + ppe_cls)
    df_output_individual = pd.DataFrame(columns=['video'] + ppe_cls)
    df_output_nonadherence = pd.DataFrame(columns=['video', 'gown', 'eyewear', 'mask', 'glove'])

    for video in videos:
        df = pd.read_excel(f'{input_exp_dir}/{video}_second.xlsx', sheet_name='Sheet1')
        df_final_output, df_interpolated = post_process(df, head=False, body=False, tail=False)
        df_final_output = df_final_output.loc[:, ~df_final_output.columns.str.contains('^Unnamed')]
        sum_individual_ppe = df_final_output.loc[:, df_final_output.columns != 'id'].sum()
        df_output_individual = compute_ape(df_output_individual, sum_individual_ppe, df_ground_truth, video)
        # df_output_nonadherence = compute_nonadherence_ape(df_output_nonadherence, sum_individual_ppe, df_ground_truth, video)
        sum_individual_ppe['video'] = video 
        sum_individual_ppe = pd.DataFrame([sum_individual_ppe])
        df_pred_individual = pd.concat([df_pred_individual, sum_individual_ppe], ignore_index=True)
        nonadherence_row = compute_nonadherence_ape(sum_individual_ppe, df_ground_truth, video)
        df_output_nonadherence = pd.concat([df_output_nonadherence, nonadherence_row], ignore_index=True)
    df_output_individual = pd.concat([df_ground_truth, df_pred_individual, df_output_individual], ignore_index=True)
    df_output_individual.to_excel(f'{output_exp_dir}/results_individual_APE.xlsx', index=False)
    df_output_nonadherence.to_excel(f'{output_exp_dir}/results_nonadherence_APE.xlsx', index=False)
    return

def post_process(df, head:bool, body:bool, tail:bool):
    df_final_output = pd.DataFrame(columns=['id'] + ppe_cls)
    df_interpolated = pd.DataFrame(columns=['id', 'second', 'gown', 'eyewear', 'mask', 'glove'])
    df_track_by_id = df.groupby('id')
    for id, subdf in df_track_by_id:
        interpolated_subdf = interpolate_na(subdf, head, body, tail)
        df_interpolated = pd.concat([df_interpolated, interpolated_subdf], ignore_index=True)

        entry = {'id': id}
        for key in ppe_cls:
            entry[key] = (interpolated_subdf == key).sum().sum()

        df_final_output = pd.concat([df_final_output, pd.DataFrame([entry])], ignore_index=True)
    return df_final_output, df_interpolated


def interpolate_na(subdf, head:bool, body:bool, tail:bool):
    for col in attr:
        col_values = subdf[col].values

        if head:
            first_non_na_index = np.where(col_values != 'na')[0]
            if len(first_non_na_index) > 0:
                first_non_na_value = col_values[first_non_na_index[0]]
                initial_na_indices = np.where(col_values == 'na')[0]

                if len(initial_na_indices) > 0 and initial_na_indices[0] < first_non_na_index[0]:
                    col_values[:first_non_na_index[0]] = first_non_na_value

        subdf[col] = col_values

        if body:
            na_indices = np.where(col_values == 'na')[0]
            if len(na_indices) == 0:
                continue

            gap_start = None
            for i in range(len(na_indices)):
                if gap_start is None:
                    gap_start = na_indices[i]
                if i == len(na_indices) - 1 or na_indices[i + 1] != na_indices[i] + 1:
                    gap_end = na_indices[i]
                    gap_length = gap_end - gap_start + 1

                    if gap_start > 0 and gap_end < len(col_values) - 1:
                        if col_values[gap_start - 1] == col_values[gap_end + 1]:
                            col_values[gap_start:gap_end + 1] = col_values[gap_start - 1]

                    gap_start = None

        subdf[col] = col_values

        if tail:            
            if subdf[col].iloc[-1] == 'na' and len(np.unique(col_values)) > 1:
                for i in range(len(subdf)-1, -1, -1):
                    if subdf[col].iloc[i] != 'na':
                        row_index_of_last_non_na = i
                        last_ppe = subdf[col].iloc[i]
                        break
                
                for i in range(row_index_of_last_non_na, len(subdf)):
                    if subdf[col].iloc[i] == 'na':
                        subdf.loc[subdf.index[i], col] = last_ppe

                else:
                    continue

    return subdf


def compute_ape(df_output_individual, sum_individual_ppe, df_ground_truth, video_name):
    gt_row = df_ground_truth[df_ground_truth['video'] == video_name]
    gt_row = gt_row.iloc[0]
    row_data = {'video': video_name}

    for col in ppe_cls:
        if gt_row[col] != 0:
            row_data[col] = abs(gt_row[col] - sum_individual_ppe[col]) / gt_row[col] * 100
        else:
            row_data[col] = None

    row_df = pd.DataFrame([row_data])
    df_output_individual = pd.concat([df_output_individual, row_df], ignore_index=True)

    return df_output_individual


def compute_nonadherence_ape(sum_individual_ppe, df_ground_truth, video_name):

    pred_nonad_gown = int(sum_individual_ppe['ga'] + sum_individual_ppe['gi'])
    pred_nonad_eye = int(sum_individual_ppe['ea'])
    pred_nonad_mask = int(sum_individual_ppe['mi'] + sum_individual_ppe['ma'])
    pred_nonad_glove = int(sum_individual_ppe['ha'])

    print('pred: ', pred_nonad_glove)

    gt_nonad_gown = int(df_ground_truth.loc[df_ground_truth['video']==video_name, 'ga'].values[0] + df_ground_truth.loc[df_ground_truth['video']==video_name, 'gi'].values[0])
    gt_nonad_eye = int(df_ground_truth.loc[df_ground_truth['video']==video_name, 'ea'].values[0])
    gt_nonad_mask = int(df_ground_truth.loc[df_ground_truth['video']==video_name, 'mi'].values[0] + df_ground_truth.loc[df_ground_truth['video']==video_name, 'ma'].values[0])
    gt_nonad_glove = int(df_ground_truth.loc[df_ground_truth['video']==video_name, 'ha'].values[0])

    print('gt: ', gt_nonad_glove)

    ape_gown = (abs(gt_nonad_gown - pred_nonad_gown) / gt_nonad_gown * 100)
    # print(ape_gown)
    ape_eyewear = (abs(gt_nonad_eye - pred_nonad_eye) / gt_nonad_eye * 100)
    ape_mask = (abs(gt_nonad_mask - pred_nonad_mask) / gt_nonad_mask * 100)
    ape_glove = (abs(gt_nonad_glove - pred_nonad_glove) / gt_nonad_glove * 100)

    nonadherence_row = pd.DataFrame([{'video': video_name, 'gown': ape_gown, 'eyewear': ape_eyewear, 'mask': ape_mask, 'glove': ape_glove}])

    return nonadherence_row

if __name__ == "__main__":
    main()