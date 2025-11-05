import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from scipy.signal import butter, filtfilt

####################################### 주요 값들 계산 (Troque / Force / Pressure) #######################################

####### 1. TORQUE #######
#Get_Motor_Torque
def get_torque(current):
    torque = 0.26813144137416156 + -0.0018624133148404981*current
    
    return torque

#Get_Interaction_AND_Gravity_Torque
def get_inter_gravi_torque(df, row_name):

    #팔길이
    length_list = [35, 35, 34, 30, 31, 37, 34, 34, 29, 31, 37, 31, 30.5, 26.5, 26]
    #몸무게
    weight_list = [70, 65, 70, 55, 48, 70, 75, 70, 60, 60, 75, 53, 58, 50, 53]
    g = 9.80665 #중력가속도
    weight_portion = 0.0187 #팔이 차지하는 무게 비율
    exo_weight = 0.6 #기기무게

    torques = []
    G_torques = []

    for i in range(1,16):
        filtered_df = df[df['Number']==i] #Number는 사용자 ID
        length = length_list[i-1]

        #inter_torque_calculating         
        weights = filtered_df[row_name]
        sin_value = np.sin(np.deg2rad(90))

        for weight in weights:
            torque = length * weight * sin_value
            torques.append(torque)

        #gravity_torque_calculating 
        G_weight = (weight_list[i-1]*weight_portion) + exo_weight
        sin_values_G = [np.sin(np.deg2rad(180-angle)) for angle in filtered_df['Angle']] #sine 계산만 됨

        for sin_value_G in sin_values_G:
            G_torque = ((length/2)*0.01) * G_weight * g * sin_value_G
            G_torques.append(G_torque)
            
    
    df['inter_torque'] = torques
    df['gravity_torque'] = G_torques

    return df

#Get_Human_Torque
def get_human_torque(df):
    
    df['Human_Torque']=df['Inter_Torque']/10197.162 -df['Gravity_Torque'] - df['Motor_Torque']

    return df

####### 2. FORCE  #######

def get_force(mass):

    g = 9.8 #중력가속도
    force = g * mass

    return force

####### 3. PRESSURE  #######

def get_pressure(df):
    
    #하지 exo 실험 후 적용예정이라 사인파만 출력중임
    num_samples = len(df)
    t = np.linspace(0, 60, num_samples, endpoint=False)
    waveform = 10 * np.sin(2 * np.pi * 20 * t)
    df['inter_pressure'] = waveform

    return df

####### 4. TREJECTORY  #######

def get_coord(length, angle):
    # 물체의 초기 위치
    x = 0
    y = 0

    # 각도를 라디안으로 변환
    angle_rad = math.radians(angle)

    # 길이와 각도를 이용하여 새로운 좌표 계산
    new_x = x + length * math.cos(angle_rad)
    new_y = y + length * math.sin(angle_rad)
    new_coord = [new_x, new_y]
    
    return new_coord


####################################### 주요 값들 전처리  ####################################### 

#######  LPF (Low pass filtering)  #######

def low_pass_filter(data, column_name, degree):

    # 샘플 주파수 및 절단 주파수 설정
    sample_frequency = 100  
    cutoff_frequency = degree  

    nyquist_frequency = 0.5 * sample_frequency
    normalized_cutoff_frequency = cutoff_frequency / nyquist_frequency

    b, a = butter(2, normalized_cutoff_frequency, btype='low', analog=False)

    data_to_filter = data[column_name].values
    filtered_data = filtfilt(b, a, data_to_filter)
    data['Filtered_'+str(column_name)] = filtered_data

    return data


#######  Rearranging by Task Sequence  #######

def re_arrange (df, count):

    new_df = pd.DataFrame()

    for i in range (1, count+1):
        filtered_df = df[df['Repeat'] == i]

        if filtered_df[filtered_df['State'] == 'Flexion'].shape[0] > 0:
            flexion_sorted_df = filtered_df[filtered_df['State'] == 'Flexion'].sort_values('Angle', ascending=False)
            new_df = pd.concat([new_df, flexion_sorted_df])
            new_df.reset_index()

        if filtered_df[filtered_df['State'] == 'Extension'].shape[0] > 0:
            extension_sorted_df = filtered_df[filtered_df['State'] == 'Extension'].sort_values('Angle', ascending=True)
            new_df = pd.concat([new_df, extension_sorted_df])
            new_df.reset_index()

    return new_df


#######  Mapping motion percent  #######

def map_motion_percent (df):
    
    motion_table_file_path = 'motion_sequence.csv'
    motion_df = pd.read_csv(motion_table_file_path)

    motion_values =[]
    
    df['Angle_int'] = df['Angle'].astype(int)

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

####################################### 최종 데이터 프레임  ####################################### 

#######  Making All Table by NUMBER of Repeat  #######

def make_all_table(folder_path, degree):

    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    merged_data = pd.DataFrame()

    for file in csv_files:
        data = pd.read_csv(file)
        print(file)
        try:
            data['Angle'] = data['Angle'].astype(int)
            data = low_pass_filter(data, 'Weight', degree) #inter_torque ->LPF by degree
            data = get_inter_gravi_torque(data, 'Filtered_Weight') 
            data = low_pass_filter(data, 'Current', degree)
            merged_data = pd.concat([merged_data, data])

        except KeyError:
            pass 

   
    grouped_data = merged_data.groupby(['State', 'Repeat', 'Angle']).agg({'Current': 'mean','Filtered_Current':'mean' ,'Weight': 'mean','Filtered_Weight': 'mean', 'inter_torque' : 'mean','gravity_torque': 'mean'}).reset_index()
    print(grouped_data)
    new_data_frame = pd.DataFrame({
        'State': grouped_data['State'],
        'Repeat': grouped_data['Repeat'],
        'Angle': grouped_data['Angle'],
        'Weight': grouped_data['Weight'],
        'Filtered_Weight': grouped_data['Filtered_Weight'],
        'Current': grouped_data['Current'],
        'Filtered_Current': grouped_data['Filtered_Current'],
        'Inter_Torque':grouped_data['inter_torque'],
        'Gravity_Torque':grouped_data['gravity_torque']})

    return new_data_frame



####################################### Main Function  ####################################### 

####### MAIN.  Make repeat table (+LPF+AVG+Torq+Current) #######

def main():
    
    #1. 폴더 -> LPF (degree) -> 평균 -> 합치기
    repeat_data = make_all_table('data/off_lift', 15)
    
    #2 sequence 별로 정리하기
    repeat_data = re_arrange(repeat_data, 6)

    #2.1 force / torque 계산하기
    repeat_data['Inter_Force'] = get_force(repeat_data['Filtered_Weight'])
    
    #2.2 motor torque 계산하기
    repeat_data['Motor_Torque'] = get_torque(repeat_data['Filtered_Current'])

    #2.3 human torque 계산하기
    repeat_data = get_human_torque(repeat_data)

    #3 파일 형태로 저장하기
    repeat_data.to_csv('result_repeat.csv', index = False)
    
    print('REPEAT TABLE SAVEDD')



if __name__ == '__main__':
    main()
    

