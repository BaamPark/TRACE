import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.colors as mcolors

def main():
    id = [3]
    file_path = "output_videos/1125_2025_swin26/1125_2025_swin26_fold4/10.29 scenario 5 trauma 1 above bed_second.xlsx"
    dataframe = pd.read_excel(file_path, sheet_name='Sheet1')
    interpolate=True
    # Create a figure for multiple timelines
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))

    # Plot all timelines in subplots781
    # plot_timeline_for_id_tp_gray(dataframe, id, 'gown', interpolate=False, tp='gc', attribute_cls_list=['na','gc', 'ga', 'gi'], ax=axes[0])
    # plot_timeline_for_id_tp_gray(dataframe, id, 'mask', interpolate=False, tp='mi', attribute_cls_list=['na', 'nc', 'mi', 'ma'], ax=axes[1])
    # plot_timeline_for_id_tp_gray(dataframe, id, 'eyewear', interpolate=False, tp='gg', attribute_cls_list=['na', 'pr', 'gg', 'fc', 'ea'], ax=axes[2])
    # plot_timeline_for_id_tp_gray(dataframe, id, 'glove', interpolate=False, tp='ha', attribute_cls_list=['na', 'hc', 'ha'], ax=axes[3])

    plot_timeline_for_id(dataframe, id, 'gown', interpolate=interpolate, tp='gc', attribute_cls_list=['na','gc', 'ga', 'gi'], ax=axes[0])
    plot_timeline_for_id(dataframe, id, 'mask', interpolate=interpolate, tp='mi', attribute_cls_list=['na', 'nc', 'mi', 'ma'], ax=axes[1])
    plot_timeline_for_id(dataframe, id, 'eyewear', interpolate=interpolate, tp='gg', attribute_cls_list=['na', 'pr', 'gg', 'fc', 'ea'], ax=axes[2])
    plot_timeline_for_id(dataframe, id, 'glove', interpolate=interpolate, tp='ha', attribute_cls_list=['na', 'hc', 'ha'], ax=axes[3])

    plt.tight_layout()
    plt.show()


def interpolate_na_values(df, max_na_to_interpolate=1000):
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

                print(col_values[:first_non_na_index[0]])
                
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

    df = interpolate_na_tail(df)
    return df


def interpolate_na_tail(df):
    df_output = pd.DataFrame(columns=df.columns)

    for id, df_id in df.groupby('id'):
        for column in ['gown', 'mask', 'eyewear', 'glove']:
            #if unique value in the column is 'na'
            # if len(df_id[column].unique()) == 1 and df_id[column].unique()[0] == 'na':
            if df_id[column].iloc[-1] == 'na' and len(df_id[column].unique()) > 1:
                for i in range(len(df_id)-1, -1, -1):
                   if df_id[column].iloc[i] != 'na':
                       row_index_of_last_non_na = i
                       last_ppe = df_id[column].iloc[i]
                       break
                   
                for i in range(row_index_of_last_non_na, len(df_id)):
                    if df_id[column].iloc[i] == 'na':
                        # print(f"interpolate with {last_ppe}")
                        df_id.loc[df_id.index[i], column] = last_ppe

            else:
                continue

        df_output = pd.concat([df_output, df_id], ignore_index=True)
    
    return df_output


def plot_timeline_for_id(dataframe, id: int, attribute: str, interpolate: bool, tp: str, attribute_cls_list: list, ax, single_id=False):
    if not single_id:
        data = dataframe[dataframe['id'].isin(id)]
        df_sorted = data.sort_values(by=['second', 'id'], ascending=[True, False])

        # Step 2: Initialize an empty dataframe to store results
        df_deduplicated = pd.DataFrame(columns=data.columns)

        # Step 3: Iterate through each row and append to the new dataframe if 'second' is unique
        for index, row in df_sorted.iterrows():
            if row['second'] not in df_deduplicated['second'].values:
                df_deduplicated = pd.concat([df_deduplicated, pd.DataFrame([row])], ignore_index=True)
        
        data = df_deduplicated
    else:
        data = dataframe[dataframe['id'] == id]
    
    if interpolate:
        data = interpolate_na_values(data)
    data['second'] = pd.Categorical(data['second'], categories=sorted(data['second'].unique()), ordered=True)
    attribute_cls_list = [cls.upper() for cls in attribute_cls_list]

    # Convert the column values in data[attribute] to uppercase
    data[attribute] = data[attribute].str.upper()

    # Update the categorical data with uppercase labels
    data[attribute] = pd.Categorical(data[attribute], categories=attribute_cls_list, ordered=True)

    # Grouping data for timeline
    timeline = data.groupby(['second', attribute]).size().unstack(fill_value=0)

    # Custom colormap logic
    custom_colormaps = []
    false_positive_attribute_cls_list = [item for item in attribute_cls_list if item != 'NA']  # Ensure 'NA' is capitalized

    for fp in false_positive_attribute_cls_list:
        if 'NA' in attribute_cls_list and attribute_cls_list.index('NA') == 0:
            viridis = plt.cm.get_cmap('viridis', len(attribute_cls_list))
            new_colors = viridis(np.linspace(0, 1, len(attribute_cls_list)))
            new_colors[0] = mcolors.to_rgba('gray')  # Set the first color (for 'NA') as gray
            custom_colormaps.append(mcolors.LinearSegmentedColormap.from_list('custom', new_colors))
        elif fp in attribute_cls_list:
            custom_colormap = plt.cm.viridis

    # Plotting the timeline on the provided axis
    timeline.plot(kind='bar', stacked=True, ax=ax, colormap=custom_colormaps[0], width=1, legend=False)

    # Formatting x-axis
    total_frames = len(timeline.index)
    positions = [i for i in range(0, total_frames, max(1, total_frames // 8))]  # Avoid division by zero
    labels = [timeline.index[i] for i in positions]

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_title(f'Timeline: {attribute} occurrence per second', fontsize=16, fontweight='bold')
    ax.set_xlabel('', fontsize=12)
    # ax.set_ylabel('OCCURRENCE', fontsize=12)

    # Setting legend with fully capitalized labels
    ax.legend(
        title=f'{attribute} TYPE', 
        labels=attribute_cls_list,  # Use fully capitalized labels in the legend
        loc='center left', 
        bbox_to_anchor=(1, 0.5), 
        prop={'size': 16, 'weight': 'bold'},  # Set font size and weight
        title_fontsize=15  # Adjust title font size if needed
    )

    ax.tick_params(axis='x', rotation=45)


def plot_timeline_for_id_tp_gray(dataframe, id: int, attribute: str, interpolate: bool, tp: str, attribute_cls_list: list, ax, single_id=False):
    if not single_id:
        data = dataframe[dataframe['id'].isin(id)]
        df_sorted = data.sort_values(by=['second', 'id'], ascending=[True, False])

        # Step 2: Initialize an empty dataframe to store results
        df_deduplicated = pd.DataFrame(columns=data.columns)

        # Step 3: Iterate through each row and append to the new dataframe if 'second' is unique
        for index, row in df_sorted.iterrows():
            if row['second'] not in df_deduplicated['second'].values:
                df_deduplicated = pd.concat([df_deduplicated, pd.DataFrame([row])], ignore_index=True)
        
        data = df_deduplicated
    else:
        data = dataframe[dataframe['id'] == id]
    
    if interpolate:
        data = interpolate_na_values(data)
    data['second'] = pd.Categorical(data['second'], categories=sorted(data['second'].unique()), ordered=True)
    attribute_cls_list = [cls.upper() for cls in attribute_cls_list]

    # Convert the column values in data[attribute] to uppercase
    data[attribute] = data[attribute].str.upper()

    # Update the categorical data with uppercase labels
    data[attribute] = pd.Categorical(data[attribute], categories=attribute_cls_list, ordered=True)

    # Grouping data for timeline
    timeline = data.groupby(['second', attribute]).size().unstack(fill_value=0)

    # Custom colormap logic
    custom_colormaps = []
    false_positive_attribute_cls_list = [item for item in attribute_cls_list if item not in ['NA', tp]]  # Ensure 'NA' is capitalized

    viridis = plt.cm.get_cmap('viridis', len(attribute_cls_list))
    new_colors = viridis(np.linspace(0, 1, len(attribute_cls_list)))

    for i, category in enumerate(attribute_cls_list):
        if category == 'NA':
            new_colors[i] = mcolors.to_rgba('gray')
        elif category == tp.upper():
            new_colors[i] = mcolors.to_rgba('green')
        elif category in false_positive_attribute_cls_list:
            new_colors[i] = mcolors.to_rgba('red')

    custom_colormaps.append(mcolors.LinearSegmentedColormap.from_list('custom', new_colors))


    # for fp in false_positive_attribute_cls_list:
    #     if 'NA' in attribute_cls_list and attribute_cls_list.index('NA') == 0:
    #         viridis = plt.cm.get_cmap('viridis', len(attribute_cls_list))
    #         new_colors = viridis(np.linspace(0, 1, len(attribute_cls_list)))
            
    #         new_colors[0] = mcolors.to_rgba('gray')  # Set the first color (for 'NA') as gray
    #         new_colors[attribute_cls_list.index(tp.upper())] = mcolors.to_rgba('green')  # Set TP as green

    #         custom_colormaps.append(mcolors.LinearSegmentedColormap.from_list('custom', new_colors))


    # Plotting the timeline on the provided axis
    timeline.plot(kind='bar', stacked=True, ax=ax, colormap=custom_colormaps[0], width=1, legend=False)

    # Formatting x-axis
    total_frames = len(timeline.index)
    positions = [i for i in range(0, total_frames, max(1, total_frames // 8))]  # Avoid division by zero
    labels = [timeline.index[i] for i in positions]

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_title(f'Timeline: {attribute} occurrence per second', fontsize=16, fontweight='bold')
    ax.set_xlabel('', fontsize=12)
    # ax.set_ylabel('OCCURRENCE', fontsize=12)

    # Setting legend with fully capitalized labels
    ax.legend(
        title=f'{attribute} TYPE', 
        labels=attribute_cls_list,  # Use fully capitalized labels in the legend
        loc='center left', 
        bbox_to_anchor=(1, 0.5), 
        prop={'size': 16, 'weight': 'bold'},  # Set font size and weight
        title_fontsize=15  # Adjust title font size if needed
    )

    ax.tick_params(axis='x', rotation=45)

if __name__ == '__main__':
    main()
