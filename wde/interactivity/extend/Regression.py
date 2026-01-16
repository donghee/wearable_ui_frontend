import numpy as np
from scipy import stats
import os
import glob
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


#################################### For Feature Extraction #############################

def _linear_slope(t, y):
    """least-squares slope of y ~ a*t + b"""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 2:
        return np.nan
    tc = t - t.mean()
    denom = np.sum(tc * tc)
    if denom <= 1e-12:
        return np.nan
    return float(np.sum(tc * (y - y.mean())) / denom)


def features_from_trial(
    trial_df: pd.DataFrame,
    eps: float = 1e-6,
    use_log_ratio: bool = False
) -> np.ndarray:
    """
    Returns feature vector:
    [
      curvature_ratio,
      slope_first,
      slope_second,
      weight_mean,
      weight_abs_max,
      weight_rms
    ]
    """
    # columns: t, angle_deg, weight
    x = trial_df["t"].to_numpy(dtype=float)
    angle = trial_df["angle_deg"].to_numpy(dtype=float)
    w = trial_df["weight"].to_numpy(dtype=float)

    # NaN 제거
    mask = np.isfinite(x) & np.isfinite(angle) & np.isfinite(w)
    x, angle, w = x[mask], angle[mask], w[mask]

    # ----- (1~3) slope-based features -----
    if len(angle) >= 6:
        n = len(angle)
        mid = n // 2

        # time 정규화 → trial 길이 차이 제거
        t = np.linspace(0.0, 1.0, n)

        slope_first  = _linear_slope(t[:mid], angle[:mid])
        slope_second = _linear_slope(t[mid:], angle[mid:])

        if np.isfinite(slope_first) and np.isfinite(slope_second):
            if use_log_ratio:
                curvature_ratio = float(
                    np.log((abs(slope_second) + eps) / (abs(slope_first) + eps))
                    * np.sign(slope_second) * np.sign(slope_first)
                )
            else:
                denom = slope_first if abs(slope_first) > eps else np.sign(slope_first) * eps + eps
                curvature_ratio = float(slope_second / denom)
        else:
            curvature_ratio = np.nan
    else:
        slope_first = np.nan
        slope_second = np.nan
        curvature_ratio = np.nan

    # ----- (4~6) weight-based features -----
    if len(w):
        weight_mean = float(np.mean(w))
        weight_abs_max = float(np.max(np.abs(w)))
        weight_rms = float(np.sqrt(np.mean(w ** 2)))
    else:
        weight_mean = np.nan
        weight_abs_max = np.nan
        weight_rms = np.nan

    return np.array(
        [
            curvature_ratio,
            slope_first,
            slope_second,
            weight_mean,
            weight_abs_max,
            weight_rms,
        ],
        dtype=float,
    )

def pairwise_feature_diffs(task1_trials, task3_trials):
    """
    task1_trials: list[pd.DataFrame]  (Task1 trials)
    task3_trials: list[pd.DataFrame]  (Task3 trials)
    return:
      D: (n, 3)  where D[i] = feat(Task3[i]) - feat(Task1[i])
      meta: trial 매칭 정보 DataFrame
    """
    n = min(len(task1_trials), len(task3_trials))
    if n == 0:
        return np.empty((0, 3)), pd.DataFrame()

    F1 = []
    F3 = []
    rows = []
    for i in range(n):
        f1 = features_from_trial(task1_trials[i])
        f3 = features_from_trial(task3_trials[i])
        
        F1.append(f1)
        F3.append(f3)
        rows.append({
            "pair_idx": i,
            "task1_file": task1_trials[i].attrs.get("file", f"task1_{i}"),
            "task3_file": task3_trials[i].attrs.get("file", f"task3_{i}"),
        })

    return pd.DataFrame(rows), np.vstack(F1), np.vstack(F3)


def compute_domain_diffs_for_rsa(data):
    """
    return:
      diffs["exp"] = (n,3)  trial별 diff 벡터
      diffs["sim"] = (n,3)
      metas["exp"], metas["sim"] = 매칭 메타
    """
    feat_names = ["curvature_ratio", "slope_first", "slope_second", "weight_mean", "weight_abs_max", "weight_rms"]

    diffs = {}
    metas = {}
    feat1 = {}
    feat3 = {}

    for domain in ["exp", "sim"]:
        meta, F1, F3 = pairwise_feature_diffs(
            data["Task1"][domain],
            data["Task3"][domain]
        )
        metas[domain] = meta
        feat1[domain] = F1
        feat3[domain] = F3

        # 보기 좋게 diff를 DF로도
        feat1_df = pd.DataFrame(F1, columns=feat_names)
        feat1[(domain, "df")] = pd.concat([meta, feat1_df], axis=1)

        feat3_df = pd.DataFrame(F3, columns=feat_names)
        feat3[(domain, "df")] = pd.concat([meta, feat3_df], axis=1)

    return metas, feat1, feat3


#################################### For Data Load ####################################

# ---------- 실험 데이터: Repeat==1 & U_Status=='Flexion'의 "첫 시작 구간"만 추출 ----------
def extract_exp_segment(df, repeat_value=1, status_value="Flexion"):
    # 조건
    cond = (df["Repeat"] == repeat_value) & (df["U_Status"].astype(str).str.strip() == status_value)

    if not cond.any():
        # 조건이 없으면 빈 DF 반환
        return pd.DataFrame(columns=["t", "angle_deg", "weight"])

    # "첫 시작부분" = 조건이 처음 True가 되는 지점부터, 연속으로 True인 구간만
    idx = np.flatnonzero(cond.values)
    start = idx[0]
    end = start
    while end + 1 < len(cond) and cond.iloc[end + 1]:
        end += 1

    seg = df.iloc[start:end+1].copy()

    out = seg[["Elbow Angle", "Weight"]].copy()
    out = out.rename(columns={"Elbow Angle": "angle_deg", "Weight": "weight"})
    out.insert(0, "t", np.arange(len(out)))
    return out


# ---------- 시뮬 데이터: sensor_position/sensor_pos + sensor_force를 표준 컬럼으로 ----------
def load_sim_standard(df):
    cols = [c.lower() for c in df.columns]

    # time column
    if "time_step" in cols:
        tcol = df.columns[cols.index("time_step")]
        t = df[tcol].values
    else:
        t = np.arange(len(df))

    # angle column 후보들
    angle_col = None
    for cand in ["sensor_pos", "sensor_position", "pos", "position"]:
        if cand in cols:
            angle_col = df.columns[cols.index(cand)]
            break

    # force column 후보들
    force_col = None
    for cand in ["sensor_force", "force", "weight"]:
        if cand in cols:
            force_col = df.columns[cols.index(cand)]
            break

    out = pd.DataFrame({"t": t})

    # angle unit heuristic:
    # 값 범위가 20보다 크면 대체로 deg(예: 0~180), 아니면 rad(예: 0~pi)
    if angle_col is not None:
        ang = df[angle_col].astype(float).values
        if np.nanmax(np.abs(ang)) > 20:     # likely degrees
            out["angle_deg"] = ang
        else:                               # likely radians
            out["angle_deg"] = np.rad2deg(ang)
    else:
        out["angle_deg"] = np.nan

    out["weight"] = df[force_col].astype(float).values if force_col is not None else np.nan
    return out


# ---------- 폴더에서 파일들 읽어서 Task별로 정리 ----------
def load_all_tasks(base_dir="."):
    # 폴더 매핑 (Task1=알고리즘1, Task3=알고리즘2)
    task_map = {
        "Task1": {"algo": 1, "exp_dir": os.path.join(base_dir, "data", "exp", "Task1"),
                          "sim_dir": os.path.join(base_dir, "data", "sim", "Task1")},
        "Task3": {"algo": 2, "exp_dir": os.path.join(base_dir, "data", "exp", "Task3"),
                          "sim_dir": os.path.join(base_dir, "data", "sim", "Task3")},
    }

    # 결과 구조: data[TaskX]["exp" or "sim"] = list of trial DataFrames
    data = {}

    for task, info in task_map.items():
        data[task] = {
            "algo": info["algo"],
            "exp": [],
            "sim": [],
        }

        # --- 실험 데이터: W로 시작 ---
        exp_files = sorted(glob.glob(os.path.join(info["exp_dir"], "W*.csv")))
        for fp in exp_files:
            df = pd.read_csv(fp)
            trial = extract_exp_segment(df)  # t, angle_deg, weight
            trial.attrs["file"] = fp
            data[task]["exp"].append(trial)

        # --- 시뮬 데이터: simulation_log로 시작 ---
        sim_files = sorted(glob.glob(os.path.join(info["sim_dir"], "simulation_log*.csv")))
        for fp in sim_files:
            df = pd.read_csv(fp)
            trial = load_sim_standard(df)    # t, angle_deg, weight
            trial.attrs["file"] = fp
            data[task]["sim"].append(trial)

    return data


# ---------- 요약 출력 ----------
def summarize_loaded(data):
    for task, d in data.items():
        print(f"\n[{task}] (Algorithm {d['algo']})")
        print(f"  EXP trials: {len(d['exp'])}")
        if len(d["exp"]) > 0:
            lens = [len(x) for x in d["exp"]]
            print(f"    EXP length (min/mean/max): {min(lens)}/{np.mean(lens):.1f}/{max(lens)}")

        print(f"  SIM trials: {len(d['sim'])}")
        if len(d["sim"]) > 0:
            lens = [len(x) for x in d["sim"]]
            print(f"    SIM length (min/mean/max): {min(lens)}/{np.mean(lens):.1f}/{max(lens)}")


############################  통계분석  ######################################
def feature_ratio(X1, X3, eps=1e-8):
    """
    X1: (N, P)  Task1 features
    X3: (N, P)  Task3 features
    return:
      R: (N, P)  = X3 / X1
    """
    X1 = np.asarray(X1, dtype=float)
    X3 = np.asarray(X3, dtype=float)

    return X3 / (X1 + eps)

def soft_clip(X, k=2.5):
    """
    tanh 기반 soft clipping
    """
    X = np.asarray(X, dtype=float)
    return k * np.tanh(X / k)

def singlevariate_regressor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    kernel: str = "rbf",
    C: float = 1e3,
    epsilon: float = 1e-5,
    gamma: str | float = 1e2,
) -> tuple[np.ndarray, Pipeline]:
    
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Row mismatch: X has {X.shape[0]} rows, y has {y.shape[0]}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)),
    ])

    model.fit(X, y)
    y_pred = model.predict(X)

    return y_pred, model

def multivariate_regressor(R_sim, R_exp, **svr_kwargs):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", MultiOutputRegressor(SVR(**svr_kwargs)))
    ])
    pipe.fit(R_sim, R_exp)
    R_hat = pipe.predict(R_sim)
    return pipe, R_hat

def ridge_regression(R_exp, R_sim, alpha=1.0):
    """
    Ridge regression: R_exp -> R_sim

    Returns
    -------
    results : dict
        {
          "W": weight matrix (P,P),
          "Y": standardized target,
          "Yhat": standardized prediction,

          "r2_overall",
          "rmse_overall",
          "mae_overall",
          "pearson_overall",
          "cosine_overall",

          "r2_per_feature",
          "rmse_per_feature",
          "pearson_per_feature",

          "W_norm",
          "X_condition"
        }
    """
    # ---------- scaling ----------
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_x.fit_transform(R_exp)
    Y = scaler_y.fit_transform(R_sim)

    # ---------- model ----------
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, Y)
    Yhat = model.predict(X)

    # ---------- overall metrics ----------
    r2_overall = r2_score(Y, Yhat)
    rmse_overall = np.sqrt(mean_squared_error(Y, Yhat))
    mae_overall = mean_absolute_error(Y, Yhat)

    pearson_overall = np.corrcoef(Y.flatten(), Yhat.flatten())[0, 1]

    # cosine similarity (directional agreement)
    num = np.sum(Y * Yhat)
    denom = np.linalg.norm(Y) * np.linalg.norm(Yhat)
    cosine_overall = num / denom if denom > 0 else np.nan

    # ---------- per-feature metrics ----------
    P = Y.shape[1]
    r2_pf = np.zeros(P)
    rmse_pf = np.zeros(P)
    pearson_pf = np.zeros(P)

    for j in range(P):
        r2_pf[j] = r2_score(Y[:, j], Yhat[:, j])
        rmse_pf[j] = np.sqrt(mean_squared_error(Y[:, j], Yhat[:, j]))
        pearson_pf[j] = np.corrcoef(Y[:, j], Yhat[:, j])[0, 1]

    # ---------- model diagnostics ----------
    W = model.coef_.T              # (P,P)
    W_norm = np.linalg.norm(W, ord="fro")
    X_condition = np.linalg.cond(X)

    return {
        "W": W,
        "Y": Y,
        "Yhat": Yhat,

        "r2_overall": r2_overall,
        "rmse_overall": rmse_overall,
        "mae_overall": mae_overall,
        "pearson_overall": pearson_overall,
        "cosine_overall": cosine_overall,

        "r2_per_feature": r2_pf,
        "rmse_per_feature": rmse_pf,
        "pearson_per_feature": pearson_pf,

        "W_norm": W_norm,
        "X_condition": X_condition,
    }

############################  Main  ############################

if __name__ == "__main__":
    
    data = load_all_tasks(base_dir=".")
    summarize_loaded(data)

    metas, feat1, feat3 = compute_domain_diffs_for_rsa(data)

    X_exp1 = feat1["exp"]
    X_sim1 = feat1["sim"]

    X_exp3 = feat3["exp"]  
    X_sim3 = feat3["sim"]  

    R_exp = soft_clip(feature_ratio(X_exp1, X_exp3))
    R_sim = feature_ratio(X_sim1, X_sim3)

    pipe, R_hat = multivariate_regressor(
        R_sim, R_exp,
        kernel="rbf",
        C=10.0,
        epsilon=0.05
    )

    res = ridge_regression(R_exp, R_hat, alpha=1.0)
    print("\n")
    print("Summary of Results")
    print("===========================================================================================================")
    print("R² (overall):", res["r2_overall"])
    print("-----------------------------------------------------------------------------------------------------------")
    print("RMSE (overall):", res["rmse_overall"])
    print("MAE (overall):", res["mae_overall"])
    print("Pearson r (overall):", res["pearson_overall"])
    print("Cosine sim (overall):", res["cosine_overall"])
    print("-----------------------------------------------------------------------------------------------------------")
    print("R² per feature:", res["r2_per_feature"])
    print("Pearson per feature:", res["pearson_per_feature"])
    print("-----------------------------------------------------------------------------------------------------------")
    print("|W|_F:", res["W_norm"])
    print("Condition number of X:", res["X_condition"])
    print("===========================================================================================================")

