import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from scipy.signal import butter, filtfilt

############### 우선 대략적으로 상호작용 점수를 구하는 순서 입니다 ###############

####### 1. 최종 데이터 가져오기 #######

def get_final_data(file_path):
    
    df = pd.read_csv(file_path)
    
    return df

####### 00. 주관식 응답 정보 아직 미정 #######

def get_survey_data(file_path):

    df_survey = df = pd.read_csv(file_path)

    return df

####### 2. 각 조건마다 점수 비교하기 #######

def compare_value(df):

    #range 에 대한 비교 파일
    df_evaluation = pd.read_csv('file for comparing.csv')

    #Angle, State마다 range 에 어느정도 들어오는지 비교하기
    #Angle, State마다 점수 매기기

    # df_input에 'Range_0', 'Range_1'을 추가한다.
    df_input = pd.merge(df_input, df_evaluation[['State', 'Angle', 'Filtered_New_Range_0', 'Filtered_New_Range_1']], 
                    on=['State', 'Angle'], how='left')

    # Human_Torque 값을 Range_0과 Range_1 사이의 상대적 위치에 따라 0에서 1 사이로 정규화한다.
    df_input['Filtered_Score'] = (df_input['Human_Torque'] - df_input['Filtered_New_Range_0']) / (df_input['Filtered_New_Range_1'] - df_input['Filtered_New_Range_0'])

    # Score 값이 0보다 작으면 0으로, 1보다 크면 1로 설정한다.
    df_input['Filtered_Score'] = df_input['Filtered_Score'].clip(0, 1)

    return df_input


####### 3. 최종 점수 구하기 #######
def main(df):

    column_values = df['Filtered_Score']  
    total_score = column_values.mean()  

    print('Interactivitiy Score is '+ str(total_score))

def test_interactivity():
    print('Interactivitiy Score is 90')
    print('- Interaction Torque: 1, 2, 1, ...')
    print('- Interaction Force: 10, 10, 10, ...')
    print('- Interaction Pressure 1, 1, 2, ...')

if __name__ == '__main__':
    main()
