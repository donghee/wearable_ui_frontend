import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from scipy.signal import butter, filtfilt

from io import BytesIO

####### 1. 최종 데이터 가져오기 #######

def get_final_data(file_path):
    
    df = pd.read_csv(file_path)
    
    return df

####### 00. Low pass filtering #######

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

###### 000. Motion 순서대로 mapping 하기 ######

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

####### 2. 각 조건마다 점수 비교하기 #######

def compare_value(df_input):

    #range 에 대한 비교 파일
    df_evaluation = pd.read_csv('Evaluation_Table2.csv')

    #Angle, State마다 점수 매기기
    # df_input에 'Range_0', 'Range_1'을 추가한다.
    df_input = pd.merge(df_input, df_evaluation[['State', 'Angle', 'New_Range_0', 'New_Range_1']], 
                    on=['State', 'Angle'], how='left')

    # Human_Torque 값을 Range_0과 Range_1 사이의 상대적 위치에 따라 0에서 1 사이로 정규화한다.
    df_input['Filtered_Score'] = (df_input['Human_Torque'] - df_input['New_Range_0']) / (df_input['New_Range_1'] - df_input['New_Range_0'])

    # Score 값이 0보다 작으면 0으로, 1보다 크면 1로 설정한다.
    df_input['Filtered_Score'] = df_input['Filtered_Score'].clip(0, 1)
    df_input.to_csv('result.csv')
    return df_input


####### 3. 입력되는 결과값 그래프 그리기 #######

#3-1. interaction force - 구해지면 출력
def inter_force_graph (df):

    inter_force = df['Inter_Force']
    plt.figure(figsize=(20,8))
    x = np.arange(len(inter_force))
    
    plt.plot(inter_force, df['Motion'], label='1',color='red')
    plt.grid(True, color = 'gray', linestyle = '--', linewidth =0.5)
    plt.show()


#3-2. interaction torque
def torques_graph (df):

    df = df.sort_values('Motion')
    plt.figure(figsize=(15,7))
    inter_torque = df['Inter_Torque']/10197.162

    plt.plot(df['Motion'],inter_torque,  label='Interaction Torque',color='green')
    plt.plot(df['Motion'], df['Human_Torque'],  label='Human Torque',color='red')
    plt.plot(df['Motion'], df['Motor_Torque'],label='Motor Torque',color='blue')

    plt.grid(True, color = 'gray', linestyle = '--', linewidth =0.5)
    plt.ylim(-3, 2)
    plt.title('Torques')
    plt.xlabel('A motion (%)')
    plt.xlabel('Torques (N.m)')
    plt.legend()
    plt.show()

#3-3. human torque
def human_torque_graph (df, df_evaluate):

    #두개의 테이블을 Motion 순서 기준으로 재정리
    df_evaluate = map_motion_percent(df_evaluate)
    df = df.sort_values('Motion')
    df_evaluate = df_evaluate.sort_values('Motion')
    
    plt.figure(figsize=(15,7))
    
    #점수매기는 범위 그래프
    plt.plot(df_evaluate['Motion'], df_evaluate['New_Range_0'], label='0',color='grey')
    plt.plot(df_evaluate['Motion'], df_evaluate['New_Range_1'], label='1',color='grey')
    
    #Human torque 그래프
    plt.plot(df['Motion'], df['Human_Torque'], label='Human',color='red')
    
    plt.grid(True, color = 'gray', linestyle = '--', linewidth =0.5)
    plt.ylim(-7, 2)
    plt.xlabel('A motion (%)')
    plt.ylabel('Human Torque (N.m)')
    plt.legend()
    plt.show()
    
def combined_torques_graph(df, df_evaluate):
    df_evaluate = map_motion_percent(df_evaluate)
    df = df.sort_values('Motion')
    df_evaluate = df_evaluate.sort_values('Motion')

    plt.figure(figsize=(15, 10))

    # 첫 번째 서브플롯 (상단)
    plt.subplot(2, 1, 1)  # 2행 1열로 배치, 첫 번째 서브플롯
    inter_torque = df['Inter_Torque'] / 10197.162
    plt.plot(df['Motion'], inter_torque, label='Interaction Torque', color='green')
    plt.plot(df['Motion'], df['Motor_Torque'], label='Motor Torque', color='blue')
    plt.plot(df['Motion'], df['Human_Torque'],  label='Human Torque',color='red')

    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.ylim(-3, 2)
    plt.xlabel('A motion (%)')
    plt.ylabel('Torques (N.m)')
    plt.legend()

    # 두 번째 서브플롯 (하단)
    plt.subplot(2, 1, 2)  # 2행 1열로 배치, 두 번째 서브플롯
    plt.plot(df_evaluate['Motion'], df_evaluate['New_Range_0'], label='0', color='grey')
    plt.plot(df_evaluate['Motion'], df_evaluate['New_Range_1'], label='1', color='grey')
    plt.plot(df['Motion'], df['Human_Torque'], label='Human Torque', color='red')

    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.ylim(-7, 2)
    plt.xlabel('A motion (%)')
    plt.ylabel('Human Torque (N.m)')
    plt.legend()

    plt.tight_layout()  # 서브플롯 간의 간격 조정
    #plt.show()

    torquegraph_img = BytesIO()
    plt.savefig(torquegraph_img, format='png', dpi=72)
    plt.clf()
    torquegraph_img.seek(0)

    return torquegraph_img


####### 4. 최종 점수 구하기 #######
def main():

    file_path = '(0724)(on_move) result_human.csv'
    evaluate_file_path = 'Evaluation_Table2.csv' 
    df = pd.read_csv(file_path)
    df_evaluate = pd.read_csv(evaluate_file_path)
    
    #N번 사람 비교
    df_number = df[df['Number'] ==10]
    df_number = compare_value(df_number)
    
    column_values = df_number['Filtered_Score']
    total_score = column_values.mean()  
    
    print('Interactivitiy Score is '+ str(total_score))
    
    #inter_force_graph (df)
    #torques_graph(df_number)
    combined_torques_graph (df_number, df_evaluate)


if __name__ == '__main__':
    main()
