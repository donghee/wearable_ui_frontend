# infer_y_score_from_result_csv.py
import joblib
import pandas as pd
import numpy as np
#import Regression as rg
from . import Regression as rg
import os

import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def infer_y_score(
    result_csv_path: str,
    model_path: str = "svr_y_score_model.joblib",
    n_steps: int = 100,
) -> float:
    df = pd.read_csv(result_csv_path)
    df.rename(columns={'step':'time_step', 'elbow_angle_rad':'sensor_position', 'inter_force':'sensor_force'}, inplace=True)

    # 1) result.csv 읽기
    #df = pd.read_csv(new_result_csv_path)

    if len(df) < n_steps:
        raise ValueError(f"result.csv has only {len(df)} rows, but n_steps={n_steps}")

    # 2) 앞 time step 100개(2초)만 자르기
    df_100 = df.iloc[:n_steps].copy()

    # 2-1) sensor_position / sensor_force → (t, angle_deg, weight) 표준화
    # load_sim_standard는 sensor_position(or sensor_pos)와 sensor_force(or force/weight)를 자동 탐색합니다. :contentReference[oaicite:7]{index=7}
    trial_std = rg.load_sim_standard(df_100)

    # 3) feature 추출 (compute_domain_diffs_for_rsa 내부에서도 결국 이걸 씁니다) :contentReference[oaicite:8]{index=8}
    feat6 = rg.features_from_trial(trial_std)  # [6,] :contentReference[oaicite:9]{index=9}
    X = feat6.reshape(1, -1)                   # (1, 6)

    # 4) 학습된 모델 로드 후 점수 예측
    model = joblib.load(model_path)
    y_score = float(model.predict(X)[0]) * 20.0 # 5점 만점에서 100점 만점으로

    return y_score

def inter_graph(result_csv_path):
    df = pd.read_csv(result_csv_path)
    df.rename(columns={'step':'time_step', 'elbow_angle_rad':'sensor_position', 'inter_force':'sensor_force'}, inplace=True)

    # elbow angle rad to degree
    df['sensor_position'] = df['sensor_position'] * (180.0 / np.pi) * -1.0

    plt.figure(figsize=(15, 10))
    plt.plot(df['time_step'], df['sensor_position'], label='Sensor Position', color='blue')

    #plt.show()

    graph_img = BytesIO()
    plt.savefig(graph_img, format='png', dpi=72)
    plt.clf()
    graph_img.seek(0)

    return graph_img

def main():
    
    y = infer_y_score("./result.csv")
    print(f"Y_score = {y:.6f}")

if __name__ == "__main__":
    main()
