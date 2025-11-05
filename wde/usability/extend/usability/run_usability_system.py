################################
#
# RUN USABILITY SYSTEM MODULE
#
# Hyungseok Oh
#
################################
#
#

import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.image as mpimg
from io import BytesIO
from PIL import Image


sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import actr


#actr.load_act_r_code("wde-usability/actr7.x/usability/system_interface.lisp")
actr.load_act_r_code("wde/usability/extend/usability/system_interface.lisp")

brain_image_path = os.path.join(os.path.dirname(__file__), 'brain_img.png')

def run_usability_system(time, task, age, force, noise):
    total_usability_value = 0
    learnability = 0
    memorability = 0
    utility = 0
    efficiency = 0
    effectiveness = 0
    performance_time = 0
    #error_value = 0
    model_output_text = None

    cog_level = "high"

    actr.reset()

    actr.load_act_r_code("wde/usability/extend/usability/1.0/model_parameters.lisp")

    if task[0] == "arm":
        actr.load_act_r_code("wde/usability/extend/usability/1.0/task/simple_arm_movement_task.lisp")
    elif task[0] == "gait":
        actr.load_act_r_code("wde/usability/extend/usability/1.0/task/simple_gait_task.lisp")



    ##setting parameters
    input_twin_age_parameter(age)
    input_task_information_parameter(task)
    input_twin_cognitive_ability_parameter(cog_level)
    input_device_force_timeline_parameter(force)
    input_twin_noise_parameter(noise)
    input_task_performance_time_parameter(time)

    setting_input_parameters_for_model()


    ##run model
    if task[0] == "arm":
        model_output_text = run_arm_model_until_time(time)
    elif task[0] == "gait":
        model_output_text = run_gait_model_until_time(time)
        

    learnability = get_value_learnability_parameter()
    memorability = get_value_memorability_parameter()
    utility = get_value_utility_parameter()
    efficiency = get_value_mental_workload_parameter()
    effectiveness = get_value_effectiveness_parameter()
    performance_time = time

    print("learnability: ", learnability)
    print("utility: ", utility)
    print("efficiency: ", efficiency)
    print("effectiveness: ", effectiveness)

    total_usability_value = (learnability + utility + efficiency + effectiveness) / 4

    #make_usability_graph([total_usability_value, learnability, utility, efficiency, effectiveness])

    return_usability_value = [math.floor(total_usability_value), [learnability, utility, efficiency, effectiveness], performance_time, get_value_ACT_R_model_output()]

    #return return_usability_value
    return return_usability_value


def make_usability_graph (usability):
    x = np.arange(len(usability))
    usability_name = ["USABILITY", "LEARNABILITY", "UTILITY", "EFFICIENCY", "EFFECTIVENESS"]

    plt.bar(x, usability)
    plt.xticks(x, usability_name)


def draw_usability_graph (time, task, age, force, noise):
    model_results = run_usability_system(time, task, age, force, noise)

    usability_values = [model_results[0], model_results[1][0], model_results[1][1], model_results[1][2], model_results[1][3]]
    brain_parts_activity = actr.current_connection.evaluate_single("get-total-brain-activity")

    return usability_figure_with_brain_data(usability_values, brain_parts_activity[1], brain_parts_activity[2])


def usability_figure(values):
    # 항목 리스트
    labels = ["TOTAL USABILITY", "LEARNABILITY", "UTILITY", "EFFICIENCY", "EFFECTIVENESS"]

    # 그래프의 크기와 여백을 설정
    fig, axs = plt.subplots(1, len(labels), figsize=(12, 2.5))
    
    # 각 항목에 대한 도너츠 모양 원형 그래프를 그립니다.
    for i, (label, value) in enumerate(zip(labels, values)):
        colors = ['crimson', 'lightgray'] if label == "TOTAL USABILITY" else ['green', 'lightgray']
        axs[i].pie([value, 100 - value], labels=[None, None], startangle=90, autopct=lambda p: '', colors=colors,
                   wedgeprops=dict(width=0.4), counterclock=False)
        axs[i].set_title(label)
        axs[i].text(0, 0, f"{value}", ha='center', va='center', color='black', fontsize=14)
        
    
    # 이미지로 저장
    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    
    return img



def usability_figure_with_brain_data(values, brain_parts_activity, brain_activity_time_series):
    # 항목 리스트
    labels = ["TOTAL USABILITY", "LEARNABILITY", "UTILITY", "EFFICIENCY", "EFFECTIVENESS"]

    # 전체 그래프 레이아웃 설정
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(nrows=2, ncols=2, left=0.1, bottom=0.15,  height_ratios=[1.2, 2.3])

    # 도너츠 원형 그래프
    axs0 = fig.add_subplot(gs[0, 1:])
    for i, (label, value) in enumerate(zip(labels, values)):
        colors = ['crimson', 'lightgray'] if label == "TOTAL USABILITY" else ['blue', 'lightgray']
        axs0.pie([value, 100 - value], labels=[None, None], startangle=90, autopct='', colors=colors,
                 wedgeprops=dict(width=0.4), counterclock=False, radius=1, center=(i * 2.7, 0))
        #axs0.set_title(label)
        axs0.text(i * 2.7, 0, f"{value}", ha='center', va='center', color='black', fontsize=14)
        axs0.text(i * 2.7, 1.4, label, ha='center', va='center', color='black', fontsize=14)  # 각 원형 그래프에 라벨 추가


    # 뇌의 4부분 활성화 정도 그래프
    axs1 = fig.add_subplot(gs[1, 0])

    brain_img = mpimg.imread(brain_image_path)
    axs1.imshow(brain_img, extent=[-1, 1, -1, 1], aspect='auto', zorder=1)
    
    cmap = plt.cm.Reds
    norm = Normalize(vmin=0, vmax=100)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # 뇌의 4부분 활성화 정도를 2D 배열로 변환
    brain_activity_2d = np.array(brain_parts_activity).reshape(2, 2)
    axs1.imshow(brain_activity_2d, cmap=cmap, norm=norm, extent=[-1, 1, -1, 1], aspect='auto', zorder=0)
    axs1.set_title("Brain Activity (4 regions)")
    axs1.axis('off')

    # 시간에 따른 뇌 활성화 정도 그래프
    axs2 = fig.add_subplot(gs[1, 1])
    # 시간과 활성화 정도 데이터 추출
    times, activities = zip(*brain_activity_time_series)
    axs2.plot(times, activities, color='green')
    axs2.set_title("Brain Activity Over Time")
    axs2.set_xlabel("Time")
    axs2.set_ylabel("Activity")

    # 이미지로 저장
    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png', dpi=72)
    img.seek(0)
    
    return img


def draw_example_img():


    img = draw_usability_graph(10, ["gait", "adaptive"], 30, 30, 5)

    # Convert BytesIO object to PIL image
    
    img_pil = Image.open(img)

    # check if the image is in RGBA format

    if img_pil.mode != 'RGBA':
        img_pil = img_pil.convert('RGBA')

    plt.close('all')

        
    # show the image!
    plt.imshow(img_pil, cmap=None)
    plt.axis('off')
    plt.show()
    
    #plt.imshow(draw_usability_graph(10, ["gait", "adaptive"], 30, 30, 5))
    #plt.show()


def get_usability_value_from_model (time, task, age, cog_level, trajectory, force, noise):
    return actr.current_connection.evaluate_single("run-the-model-until-time", time, task, age, cog_level, trajectory, force, noise)
    



def output_total_model_parameters ():
    return actr.current_connection.evaluate_single("output-total-model-parameters")

def run_gait_model_until_time (time):

    init_utility_test("gait")

    window = actr.open_exp_window("GAIT", visible=False)
    actr.add_text_to_exp_window(window, "O", x=50, y=50)
    actr.add_text_to_exp_window(window, "X", x=50, y=75)

    actr.install_device(window)

    actr.run(time, real_time=False)

    return output_total_model_parameters()


def run_arm_model_until_time (time):

    init_utility_test("arm")

    #actr.install_device(window)

    actr.run(time, real_time=False)

    return output_total_model_parameters()



def init_utility_test (task):
    if task == "arm":
        actr.current_connection.evaluate_single("init-arm-test")
    elif task == "gait":
        actr.current_connection.evaluate_single("init-gait-test")


def setting_input_parameters_for_model ():
    return actr.current_connection.evaluate_single("setting-input-parameters")


def input_twin_age_parameter (value):
    return actr.current_connection.evaluate_single("input-twin-age-parameter", value)

def get_value_twin_age ():
    return actr.current_connection.evaluate_single("get-value-twin-age")



def input_task_information_parameter (value):
    return actr.current_connection.evaluate_single("input-task-information-parameter", value)

def get_value_task_information ():
    return actr.current_connection.evaluate_single("get-value-task-information")



def input_twin_cognitive_ability_parameter (value):
    return actr.current_connection.evaluate_single("input-twin-cognitive-ability-parameter", value)

def get_value_twin_cognitive_ability ():
    return actr.current_connection.evaluate_single("get-value-twin-cognitive-ability")
        


def input_device_force_timeline_parameter (value):
    return actr.current_connection.evaluate_single("input-device-force-timeline-parameter", value)

def get_value_device_force_timeline ():
    return actr.current_connection.evaluate_single("get-value-device-force-timeline")



def input_task_performance_time_parameter (value):
    return actr.current_connection.evaluate_single("input-task-performance-time-parameter", value)

def get_value_task_performance_time ():
    return actr.current_connection.evaluate_single("get-value-task-performance-time")



def input_twin_noise_parameter (value):
    return actr.current_connection.evaluate_single("input-twin-noise-parameter", value)

def get_value_twin_noise ():
    return actr.current_connection.evaluate_single("get-value-twin-noise")


##
##  OUTPUT PARAMETERS
##

def input_ACT_R_model_output (value):
    return actr.current_connection.evaluate_single("input-ACT-R-model-output", value)

def get_value_ACT_R_model_output ():
    return actr.current_connection.evaluate_single("get-value-ACT-R-model-output")



def input_mental_workload_parameter (value):
    return actr.current_connection.evaluate_single("input-mental-workload-parameter", value)

def get_value_mental_workload_parameter ():
    return actr.current_connection.evaluate_single("get-value-mental-workload-parameter")



def input_human_error_parameter (value):
    return actr.current_connection.evaluate_single("input-human-error-parameter", value)

def get_value_human_error_parameter ():
    return actr.current_connection.evaluate_single("get-value-human-error-parameter")



def input_learnability_parameter (value):
    return actr.current_connection.evaluate_single("input-learnability-parameter", value)

def get_value_learnability_parameter ():
    return actr.current_connection.evaluate_single("get-value-learnability-parameter")



def input_memorability_parameter (value):
    return actr.current_connection.evaluate_single("input-memorability-parameter", value)

def get_value_memorability_parameter ():
    return actr.current_connection.evaluate_single("get-value-memorability-parameter")



def input_utility_parameter (value):
    return actr.current_connection.evaluate_single("input-utility-parameter", value)

def get_value_utility_parameter ():
    return actr.current_connection.evaluate_single("get-value-utility-parameter")




def input_effectiveness_parameter (value):
    return actr.current_connection.evaluate_single("input-effectiveness-parameter", value)

def get_value_effectiveness_parameter ():
    return actr.current_connection.evaluate_single("get-value-effectiveness-parameter")



def input_total_usability_parameter (value):
    return actr.current_connection.evaluate_single("input-total-usability-parameter", value)

def get_value_total_usability_parameter ():
    return actr.current_connection.evaluate_single("get-value-total-usability-parameter")
