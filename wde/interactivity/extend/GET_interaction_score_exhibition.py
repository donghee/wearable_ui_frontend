import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from scipy.signal import butter, filtfilt

from io import BytesIO

def map_motion_percent (df):
    
    motion_table_file_path = os.path.join(os.path.dirname(__file__), 'motion_sequence2.csv')
    motion_df = pd.read_csv(motion_table_file_path)
    
    motion_df['State'] = motion_df['State'].apply(lambda x: x.lower())

    motion_values =[]
    
    df['Angle_int'] = df['Angle'].astype(int)
    df['Angle_int'] = 180 - df['Angle_int'] 

    # 데이터프레임1과 데이터프레임2를 동시에 비교
    for _, row1 in df.iterrows():
        angle1 = row1['Angle_int']
        state1 = row1['State']
    
        matching_row = motion_df[(motion_df['Angle'] == angle1) & (motion_df['State'] == state1)]
        if not matching_row.empty:
            motion_value = matching_row['Motion'].values[0]
            motion_values.append(motion_value)
        else: 
            motion_value = np.nan
            motion_values.append(motion_value)


    df['Motion'] = motion_values
    df['Motion'] = df['Motion'].fillna((df['Motion'].ffill() + df['Motion'].bfill()) / 2)

    return df

def low_pass_filter(data, column_name, degree):

    # 샘플 주파수 및 절단 주파수 설정
    sample_frequency = 100  
    cutoff_frequency = degree  

    nyquist_frequency = 0.5 * sample_frequency
    normalized_cutoff_frequency = cutoff_frequency / nyquist_frequency

    b, a = butter(2, normalized_cutoff_frequency, btype='low', analog=False)

    data_to_filter = data[column_name].values
    filtered_data = filtfilt(b, a, data_to_filter)
    data[str(column_name)] = filtered_data

    return data

def combined_torques_graph(df, df_evaluate):

    df = low_pass_filter(df, 'InteractionTorque', 30)
    df = low_pass_filter(df, 'MotorTorque', 30)
    df = low_pass_filter(df, 'HumanTorque', 30)
    
    plt.figure(figsize=(20, 15))

    # 첫 번째 서브플롯 (상단)
    plt.subplot(2, 1, 1)  # 2행 1열로 배치, 첫 번째 서브플롯
    
    plt.plot(df['Motion'], df['InteractionTorque'], label='Interaction Torque', color='green')
    plt.plot(df['Motion'], df['MotorTorque'], label='Motor Torque', color='blue')
    plt.plot(df['Motion'], df['HumanTorque'],  label='Human Torque',color='red')

    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    #plt.ylim(-3, 2)
    plt.xlabel('A motion (%)', fontsize=20)
    plt.ylabel('Torques (N.m)', fontsize=20)
    plt.legend(fontsize = 20)

    # 두 번째 서브플롯 (하단)
    plt.subplot(2, 1, 2)  # 2행 1열로 배치, 두 번째 서브플롯
    plt.plot(df_evaluate['Motion'], df_evaluate['New_Range_0'], label='0', color='grey')
    plt.plot(df_evaluate['Motion'], df_evaluate['New_Range_1'], label='1', color='grey')
    plt.plot(df['Motion'], df['HumanTorque'], label='Human Torque', color='red')

    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.ylim(-7, 20)
    plt.xlabel('A motion (%)', fontsize=20)
    plt.ylabel('Human Torque (N.m)', fontsize=20)
    plt.legend(fontsize = 20)

    plt.tight_layout()  # 서브플롯 간의 간격 조정
    #plt.show()

    torquegraph_img = BytesIO()
    plt.savefig(torquegraph_img, format='png', dpi=72)
    plt.clf()
    torquegraph_img.seek(0)

    return torquegraph_img
    
    
def compare_value(df): 
    df['HumanTorque'] = df['HumanTorque'].abs()
    mean_value = df['HumanTorque'].mean()
    
    new_min = 0
    new_max = 4

    normalized_value = (mean_value - new_min) / (new_max - new_min)
    
    return normalized_value*100

def main():

    file_path = 'interaction_control_node_1.csv'
    df = pd.read_csv(file_path)
    df = map_motion_percent(df)
    df = df.sort_values('Motion')

    evaluate_file_path = 'Evaluation_Table2.csv' 
    df_evaluate = pd.read_csv(evaluate_file_path)
    
    total_score = compare_value(df)
    
    print('Interactivitiy Score is '+ str(total_score))
    
    combined_torques_graph (df, df_evaluate)
    
if __name__ == '__main__':
    main()