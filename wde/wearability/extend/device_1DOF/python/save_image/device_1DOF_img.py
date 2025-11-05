import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from math import sin, cos, radians, degrees, pi
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from scipy.io import loadmat
import multiprocessing


# Clear console, variables, and close all plots (clc, clear, close all)
import sys
from IPython import get_ipython

# Start timer (tic)
import time
start_time = time.time()

# Define the list of folders and number of cases
folders = ['force', 'posture', 'score']
numCases = 5

# Create the 'result' directory if it doesn't exist

current_folder = os.getcwd()

if not os.path.exists('result'):
    os.mkdir('result')
else:
    shutil.rmtree('result','s')
    os.mkdir('result')

for i in range(5):
    for j in ['force', 'posture', 'score']:
        path = f'result\\case{i}\\{j}'
        os.makedirs(path, exist_ok=True)


# Read and process the input data

input_df = pd.read_csv('input.csv')
input_angle = np.round(input_df['Angle'])
input_length = len(input_angle)

input_time = pd.to_datetime(input_df['Time'], format='%H:%M:%S.%f')

start_input_time = input_time.min()
end_input_time = input_time.max()

duration_time = (end_input_time - start_input_time).total_seconds()
duration_time_matrix = (input_time - start_input_time).dt.total_seconds()



def case(m):

    # Generate file paths
    ca = f'case{m - 1}.txt'
    sc = f'{current_folder}/result/case{m - 1}/score.txt'

    # Open and write to the 'score.txt' file
    ID= open(sc,"w+")
    a = 0
        
    # Read data from the 'case?.txt' file
    file = open(ca,"r")
    G = np.loadtxt(file)
        
    for n in input_angle:

        a += 1
        posture_folder = f'{current_folder}/result/case{m - 1}/posture/'
        force_folder = f'{current_folder}/result/case{m - 1}/force/'
        score_folder = f'{current_folder}/result/case{m - 1}/score/'

        posture = os.path.join(posture_folder, f'posture{a}.jpg')
        force = os.path.join(force_folder, f'force{a}.jpg')
        score = os.path.join(score_folder, f'score{a}.jpg')

        device_baselink_angle = 90
        device_elbow_angle = n

        # constant value 73kg, 1741mm, male

        L_h_u = 0.2817 # human upper arm length[m]
        L_h_f = 0.2689 # human forearm length[m]
        L_d_u = 0.2 # device upper arm length[m]
        L_d_f = 0.2 # device forarm length[m]

        m_h_u = 1.9783 # human upper arm weight[kg]
        m_h_f = 1.1826 # human forearm weight[kg]
        m_d_u = 2 # device upper arm weight[kg]
        m_d_f = 2 # device forearm weight[kg]

        G_h_u = 0.4228 # human upper arm gravity ratio
        G_h_f = 0.4574 # human forearm gravity ratio
        G_d_u = 0.5 # robot upper arm gravity ratio
        G_d_f = 0.5 # robot forearm gravity ratio

        K_u = 10000 # human upper arm shear stiffness[N/m]
        K_f = 10000 # human forearm shear stiffness[N/m]
        K_t_u = 10000 # human upper arm torsion stiffness[N/m]
        K_t_f = 10000 # human forearm torsion stiffness[N/m]
        q_u_0 = 0 # initial value of human upper arm deg[deg]
        q_f_0 = 0 # initial value of human forearm deg[deg]

        g = 9.81 # gravitational acceleration[m/s^2]

        l_s_u_0 = L_h_u - L_d_u  # initial value of human upper arm spring[m] 
        l_s_f_0 = L_h_f - L_d_f  # initial value of human forearm srping[m]

        q_u = radians(G[np.where((G[:, 0] == device_baselink_angle) & (G[:, 1] == device_elbow_angle))[0][0], 3])
        q_f = radians(G[np.where((G[:, 0] == device_baselink_angle) & (G[:, 1] == device_elbow_angle))[0][0], 4])

        d_d = radians(device_elbow_angle)       

        l_s_u = (L_h_u*sin(q_f - d_d + q_u) + L_d_f*sin(q_f) + L_d_u*sin(d_d - q_f))/sin(q_f - d_d + q_u)
        l_s_f = (L_h_f*sin(q_f - d_d + q_u) + L_d_u*sin(q_u) + L_d_f*sin(d_d - q_u))/sin(q_f - d_d + q_u)

        h_d = d_d - q_u - q_f

        baselink_angle = radians(device_baselink_angle)


        # calculate safety socre

        Mu = K_t_u*q_u
        Mf = K_t_f*q_f
        Fu = K_u*(l_s_u - l_s_u_0)
        Ff = K_u*(l_s_f - l_s_f_0)
        AD = degrees(h_d) - degrees(d_d)

        maximum_Fu = 150
        maximum_Mu = 100
        maximum_Ff = 150
        maximum_Mf = 100
        maximum_AD = 5

        maximum_F = max([maximum_Fu, maximum_Ff])
        maximum_M = max([maximum_Mu, maximum_Mf])

        if(abs(Fu)>maximum_Fu):
            SFu = 10**(-10)
        else:
            SFu = (-1/maximum_Fu)*abs(Fu)+1

        if(abs(Mu)>maximum_Mu):
            SMu = 10**(-10)
        else:
            SMu = (-1/maximum_Mu)*abs(Mu)+1

        if(abs(Ff)>maximum_Ff):
            SFf = 10**(-10)
        else:
            SFf = (-1/maximum_Ff)*abs(Ff)+1

        if(abs(Mf)>maximum_Mf):
            SMf = 10**(-10)
        else:
            SMf = (-1/maximum_Mf)*abs(Mf)+1

        if(abs(AD)>maximum_AD):
            SA = 10**(-10)
        else:
            SA = (-1/maximum_AD)*abs(AD)+1

        St  = SFu*SMu*SFf*SMf*SA
        if(St<=10**(-10)):
            Stotal = 0
        else:
            Stotal = 0.1*(10 + np.log(St))


        # figure 1

        x = (L_d_f*cos(baselink_angle + q_u)*sin(q_f) + L_d_u*sin(d_d - q_f)*cos(baselink_angle + q_u) + L_d_u*sin(q_f - d_d + q_u)*cos(baselink_angle))/sin(q_f - d_d + q_u)
        y = (L_d_f*sin(baselink_angle + q_u)*sin(q_f) + L_d_u*sin(d_d - q_f)*sin(baselink_angle + q_u) + L_d_u*sin(q_f - d_d + q_u)*sin(baselink_angle))/sin(q_f - d_d + q_u)

        x_d_u = L_d_u * cos(baselink_angle + 0) 
        y_d_u = L_d_u * sin(baselink_angle + 0) 

        x_d_f = L_d_f * cos(baselink_angle + d_d)
        y_d_f = L_d_f * sin(baselink_angle + d_d)

        x_h_f = x + L_h_f * cos(baselink_angle + q_u + h_d)
        y_h_f = y + L_h_f * sin(baselink_angle + q_u + h_d)

        x_h_u = x + L_h_u * cos(baselink_angle + q_u)
        y_h_u = y + L_h_u * sin(baselink_angle + q_u)

        txt1 = 'device elbow angle : ' + str(round(degrees(d_d),4)) + '°'
        txt2 = 'human elbow angle : ' + str(round(degrees(h_d),4)) + '°'

        plt.title('Posture')
        plt.grid(True)
        plt.xlabel('X[m]')
        plt.ylabel('Y[m]')
        plt.xticks(np.arange(-0.5, 0.5, 0.1))
        plt.xlim([-0.5, 0.5])
        plt.yticks(np.arange(-0.4, 0.4, 0.1))
        plt.ylim([-0.4, 0.4])
        plt.plot([0, x_d_f], [0, y_d_f], color='black', linewidth=1)
        plt.plot([0, x_d_u], [0, y_d_u], color='black', linewidth=1)
        plt.plot([x, x_h_f], [y, y_h_f], color='green', linewidth=1)
        plt.plot([x, x_h_u], [y, y_h_u], color='green', linewidth=1)
        plt.plot(x, y, 'g.', markersize=15)
        plt.plot(x_d_f, y_d_f, 'g.', markersize=15)
        plt.plot(x_d_u, y_d_u, 'g.', markersize=15)
        plt.plot(0, 0, 'k.', markersize=15)
        plt.text(-0.49,-0.34,txt1)
        plt.text(-0.49,-0.39,txt2)
        plt.savefig(posture)
        plt.close()

        # figure 2

        fig2, ax1 = plt.subplots()
        plt.title('Interaction Force')
        plt.grid(True)
        ax1.set_ylabel('moment[Nm]', color = 'red')
        ax1.set_ylim([-maximum_M, maximum_M])
        ax2 = ax1.twinx()
        ax2.set_ylabel('force[N]', color = 'blue')
        ax2.set_ylim([-maximum_F, maximum_F])
        x1 = np.arange(4)
        X1 = ['Mu', 'Fu', 'Mf', 'Ff']
        Y1 = [Mu, Fu, Mf, Ff]
        C1 = ['red', 'blue', 'red', 'blue']
        bar = plt.bar(x1, Y1, color=C1)
        plt.xticks(x1, X1)

        for rect in bar:
            height = rect.get_height()
            if height > 0:
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.4f' % height, ha='center', va='bottom')
            else:
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.4f' % height, ha='center', va='top')
        plt.savefig(force)
        plt.close()


        # figure 3

        df = pd.DataFrame({
        'Safety': ['0'],
        'SMu': [SMu],
        'SFu': [SFu],
        'SMf': [SMf],
        'SFf': [SFf],
        'SA': [SA]
        })

        labels = df.columns[1:]
        num_labels = len(labels)
            
        angles = [x/float(num_labels)*(2*pi) for x in range(num_labels)] ## 각 등분점
        angles += angles[:1] ## 시작점으로 다시 돌아와야하므로 시작점 추가
            
        my_palette = plt.matplotlib.colormaps.get_cmap("Set2")

        txtscore = 'total safety score : ' + str(round(2,4))

        fig = plt.figure()
        fig.set_facecolor('white')
        ax = fig.add_subplot(polar=True)
        for i, row in df.iterrows():
            color = my_palette(i)
            data = df.iloc[i].drop('Safety').tolist()
            data += data[:1]
            
            ax.set_theta_offset(pi / 2) ## 시작점
            ax.set_theta_direction(-1) ## 그려지는 방향 시계방향
            
            plt.xticks(angles[:-1], labels) ## 각도 축 눈금 라벨
            ax.tick_params(axis='x', which='major') ## 각 축과 눈금 사이에 여백을 준다.

            ax.set_rlabel_position(0) ## 반지름 축 눈금 라벨 각도 설정(degree 단위)
            plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],['0.0','0.2','0.4','0.6','0.8','1.0']) ## 반지름 축 눈금 설정
            plt.ylim(0,1.0)
            
            ax.plot(angles, data, color=color, linewidth=2, linestyle='solid', label=row.Safety) ## 레이더 차트 출력
            ax.fill(angles, data, color=color, alpha=0.4) ## 도형 안쪽에 색을 채워준다.
        plt.title('Safety Score')
        txtscore = 'total safety score : ' + str(round(Stotal,4))
        plt.suptitle(txtscore, y=0.05, fontsize = 10)  
        plt.savefig(score)
        plt.close()

        # Print progress
        if m == 1:
            percent = f'{20 * a / input_length:.2f}%'
            print(percent)
        elif m == 2:
            percent = f'{20 + 20 * a / input_length:.2f}%'
            print(percent)            
        elif m == 3:
            percent = f'{40 + 20 * a / input_length:.2f}%'
            print(percent)
        elif m == 4:
            percent = f'{60 + 20 * a / input_length:.2f}%'
            print(percent)
        elif m == 5:
            percent = f'{80 + 20 * a / input_length:.2f}%'
            print(percent)

        ID.write(f'{Stotal}\n')
    ID.close()

def G():
    G0 = np.loadtxt(current_folder + '/result/case0/score.txt')
    G1 = np.loadtxt(current_folder + '/result/case1/score.txt')
    G2 = np.loadtxt(current_folder + '/result/case2/score.txt')
    G3 = np.loadtxt(current_folder + '/result/case3/score.txt')
    G4 = np.loadtxt(current_folder + '/result/case4/score.txt')

    All = [G0, G1, G2, G3, G4]

    # Calculate mean scores
    total_score_case0 = np.mean(G0)
    total_score_case1 = np.mean(G1)
    total_score_case2 = np.mean(G2)
    total_score_case3 = np.mean(G3)
    total_score_case4 = np.mean(G4)

    # Calculate the total score
    total_scores = [total_score_case0, total_score_case1, total_score_case2, total_score_case3, total_score_case4]
    total_score = np.mean(total_scores)
    minimum_value = np.argmin(total_scores)

    # Create a data table and save it to a CSV file
    data = {
        'Time': input_time.astype(str),
        'Angle': input_angle,
        'Score': All[minimum_value]
    }
    df = pd.DataFrame(data)
    df.to_csv(current_folder + '/result/output_expression.csv', index=False)


    # fig4 text

    txtc0 = 'safety score case0: ' + str(round(total_score_case0,4)) 
    txtc1 = 'safety score case1: ' + str(round(total_score_case1,4))
    txtc2 = 'safety score case2: ' + str(round(total_score_case2,4))
    txtc3 = 'safety score case3: ' + str(round(total_score_case3,4))
    txtc4 = 'safety score case4: ' + str(round(total_score_case4,4))
    txtto = 'total safety score : ' + str(round(total_score,4))
    txtwc = 'worst case : ' + str(minimum_value - 1)

    # figure 4
    fig4 = plt.figure()
    plt.title('Total Safety Score')
    plt.xlabel('time[sec]')
    plt.ylabel('safety score')
    plt.xticks(range(math.ceil(duration_time) + 1))
    plt.xlim(0, math.ceil(duration_time))
    plt.yticks(np.arange(0, 1.2, 0.1))
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.plot(duration_time_matrix, G0, color='black', label='Solid', linewidth=2)
    plt.plot(duration_time_matrix, G1, color='red', label='Solid', linewidth=2)
    plt.plot(duration_time_matrix, G2, color='green', label='Solid', linewidth=2)
    plt.plot(duration_time_matrix, G3, color='blue', label='Solid', linewidth=2)
    plt.plot(duration_time_matrix, G4, color='magenta', label='Solid', linewidth=2)
    plt.text(0.1,0.26,txtc0)
    plt.text(0.1,0.21,txtc1)
    plt.text(0.1,0.16,txtc2)
    plt.text(0.1,0.11,txtc3)
    plt.text(0.1,0.06,txtc4)
    plt.text(0.1,0.01,txtto, weight='bold')
    plt.text(0.1,0.31,txtwc, color='red')

    plt.savefig(current_folder + '/result/total_safety.jpg')

    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Time elapsed: {elapsed_time:.2f} seconds')


def image():
    for m in range(1,6):
        case(m)
    G()

image()