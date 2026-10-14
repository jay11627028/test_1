# -*- coding: utf-8 -*-
"""
AnalysisStatics.py

整合 Constitution、BDI-II、PSQI 三個 Statistics 分析流程。
直接讀取 Basic 階段輸出的 RecordOriginal_3_sets.pkl。
pkl 內同時包含 Valid=True 與 Valid=False，Statistics 會保留全部資料做 validation summary，統計/分群只使用 Valid=True。

執行：
    python AnalysisStatics.py

主要輸出：
    AnalysisStatics_Data/
        Constitution/
        BDI/
        PSQI/
        Integrated/
        AnalysisStatics.md
"""

import os
import pickle
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PKL = os.path.join(SCRIPT_DIR, "RecordOriginal_3_sets.pkl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "AnalysisStatics_Data")
# 分群設定：Constitution 維持 4 群；BDI / PSQI 依本資料重新評估後改為 2 群。
# 若你已經有舊輸出檔，建議設 True，避免沿用舊的 4 群結果。
OVERWRITE_OUTPUT = True
CON_N_CLUSTERS = 4
BDI_N_CLUSTERS = 2
PSQI_N_CLUSTERS = 2

AGE_BINS = [0, 19, 29, 39, 49, 59, 69, 200]
AGE_LABELS = ["<=19", "20-29", "30-39", "40-49", "50-59", "60-69", ">=70"]
SEX_MAP = {0: "Female", 1: "Male"}

CON_TYPE_NAMES = [
    "Type-1（平和體質）", "Type-2（氣虛體質）", "Type-3（陽虛體質）",
    "Type-4（陰虛體質）", "Type-5（痰濕體質）", "Type-6（濕熱體質）",
    "Type-7（血瘀體質）", "Type-8（氣鬱體質）", "Type-9（特稟體質）",
]
CON_TYPE_SHORT_NAMES = ["1.平和", "2.氣虛", "3.陽虛", "4.陰虛", "5.痰濕", "6.濕熱", "7.血瘀", "8.氣鬱", "9.特稟"]
CON_RAW_CLUSTER_INFO = {
    3: {"排序": 1, "Cluster": "Cluster-1", "Stage": "Stage-1", "名稱": "純平和", "PCA位置": "左側", "特徵": "健康"},
    1: {"排序": 2, "Cluster": "Cluster-2", "Stage": "Stage-2", "名稱": "過渡混合", "PCA位置": "中央", "特徵": "輕度偏頗"},
    2: {"排序": 3, "Cluster": "Cluster-3A", "Stage": "Stage-3A", "名稱": "全面失衡", "PCA位置": "右側", "特徵": "多體質高"},
    4: {"排序": 4, "Cluster": "Cluster-3B", "Stage": "Stage-3B", "名稱": "氣鬱濕熱", "PCA位置": "右下", "特徵": "情緒/濕熱"},
}
CON_CLUSTER_COLORS = {"Cluster-1": "tab:green", "Cluster-2": "tab:blue", "Cluster-3A": "tab:orange", "Cluster-3B": "tab:red"}

BDI_CUTOFF_MILD = 14
BDI_CUTOFF_MODERATE = 20
BDI_CUTOFF_SEVERE = 29
BDI_ITEM_COLS = [
    "Q1_Sadness", "Q2_Pessimism", "Q3_PastFailure", "Q4_LossOfPleasure", "Q5_GuiltyFeelings",
    "Q6_PunishmentFeelings", "Q7_SelfDislike", "Q8_SelfCriticalness", "Q9_SuicidalThoughts", "Q10_Crying",
    "Q11_Agitation", "Q12_LossOfInterest", "Q13_Indecisiveness", "Q14_Worthlessness", "Q15_LossOfEnergy",
    "Q16_ChangesInSleep", "Q17_IrritabilityFatigue", "Q18_ChangesInAppetite", "Q19_ConcentrationWeight",
    "Q20_TirednessHealth", "Q21_LossOfSexInterest",
]
BDI_ITEM_LABELS = {
    "Q1_Sadness": "Q1 悲傷", "Q2_Pessimism": "Q2 悲觀", "Q3_PastFailure": "Q3 過去失敗", "Q4_LossOfPleasure": "Q4 失去愉悅",
    "Q5_GuiltyFeelings": "Q5 罪惡感", "Q6_PunishmentFeelings": "Q6 受罰感", "Q7_SelfDislike": "Q7 不喜歡自己", "Q8_SelfCriticalness": "Q8 自我批評",
    "Q9_SuicidalThoughts": "Q9 自殺意念", "Q10_Crying": "Q10 哭泣", "Q11_Agitation": "Q11 易怒/激動", "Q12_LossOfInterest": "Q12 失去興趣",
    "Q13_Indecisiveness": "Q13 猶豫不決", "Q14_Worthlessness": "Q14 無價值感/外表", "Q15_LossOfEnergy": "Q15 失去活力",
    "Q16_ChangesInSleep": "Q16 睡眠改變", "Q17_IrritabilityFatigue": "Q17 疲倦", "Q18_ChangesInAppetite": "Q18 食慾改變",
    "Q19_ConcentrationWeight": "Q19 體重下降", "Q20_TirednessHealth": "Q20 健康憂慮", "Q21_LossOfSexInterest": "Q21 性興趣降低",
}
BDI_COGNITIVE_AFFECTIVE_COLS = BDI_ITEM_COLS[:14]
BDI_SOMATIC_COLS = BDI_ITEM_COLS[14:]
BDI_FEATURE_COLS = list(OrderedDict.fromkeys(["BDI_Total", "CognitiveAffective_Score", "Somatic_Score", "Q9_SuicidalThoughts"] + BDI_ITEM_COLS))

PSQI_CUTOFF = 5
PSQI_COMPONENT_COLS = ["C1_SleepQuality", "C2_SleepLatency", "C3_SleepDuration", "C4_SleepEfficiency", "C5_SleepDisturbance", "C6_SleepMedicine", "C7_DaytimeDysfunction"]
PSQI_COMPONENT_LABELS = {
    "C1_SleepQuality": "C1 睡眠品質", "C2_SleepLatency": "C2 入睡時間", "C3_SleepDuration": "C3 睡眠時間", "C4_SleepEfficiency": "C4 睡眠效率",
    "C5_SleepDisturbance": "C5 睡眠困擾", "C6_SleepMedicine": "C6 安眠藥物", "C7_DaytimeDysfunction": "C7 日間功能",
}
PSQI_FEATURE_COLS = ["PSQI_Total", "Q2_SleepLatency_Min", "ActualSleepHours", "TimeInBedHours", "SleepEfficiencyPercent"] + PSQI_COMPONENT_COLS

# PSQI 分群使用較穩定的 component-based features，避免連續睡眠時數/效率欄位主導分群。
PSQI_CLUSTER_FEATURE_COLS = ["PSQI_Total"] + PSQI_COMPONENT_COLS

GENERIC_CLUSTER_COLORS = {
    "Cluster-1": "tab:green",
    "Cluster-2": "tab:blue",
    "Cluster-3": "tab:orange",
    "Cluster-4": "tab:red",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def setup_chinese_font():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    candidates = ["Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans CJK JP", "SimHei", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", category=UserWarning)


def safe_number(value):
    if value in [None, ""]:
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def count_percent(count, total, digits=0):
    if total == 0:
        return "0%"
    return f"{count / total * 100:.{digits}f}%"


def save_if_needed(full_path, save_func, label):
    if OVERWRITE_OUTPUT or not os.path.exists(full_path):
        save_func(full_path)
        print(f"生成檔案({label}): {full_path}")
    else:
        print(f"已存在，略過生成({label}): {full_path}")


def load_integrated_valid_data(pkl_path=INPUT_PKL):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"找不到輸入 pkl: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    for key in ["Constitution", "BDI", "PSQI"]:
        if key not in data:
            raise KeyError(f"pkl 缺少必要 key: {key}")
    return data


def valid_records(data_dict):
    return OrderedDict((pid, rec) for pid, rec in data_dict.items() if bool(rec.get("Valid", False)))


def build_cons_df(cons_data):
    rows = []
    for person_id, person_data in cons_data.items():
        row = {"ID": person_data.get("ID", person_id), "Sex": person_data.get("Sex"), "Age": person_data.get("Age"), "Valid": bool(person_data.get("Valid", False)), "Invalid_Reasons": "; ".join(person_data.get("Invalid_Reasons", []) or []), "Invalid_Fields": "; ".join(person_data.get("Invalid_Fields", []) or [])}
        for type_name in CON_TYPE_NAMES:
            type_data = person_data.get(type_name, {}) or {}
            row[f"{type_name}_Score"] = safe_number(type_data.get("Score"))
            row[f"{type_name}_Ans"] = type_data.get("Ans")
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)
    df["SexLabel"] = df["Sex"].map(SEX_MAP).fillna("Unknown")
    df["AgeGroup"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=True, include_lowest=True)
    return df


def build_bdi_df(bdi_data):
    rows = []
    for person_id, person_data in bdi_data.items():
        item_scores = person_data.get("BDI_Item_Scores", {}) or {}
        risk_flags = person_data.get("Risk_Flags", {}) or {}
        row = {"ID": person_data.get("ID", person_id), "Sex": person_data.get("Sex"), "Age": person_data.get("Age"), "Valid": bool(person_data.get("Valid", False)), "Invalid_Reasons": "; ".join(person_data.get("Invalid_Reasons", []) or []), "Invalid_Fields": "; ".join(person_data.get("Invalid_Fields", []) or []), "BDI_Total": safe_number(person_data.get("BDI_Total")), "BDI_Severity": person_data.get("BDI_Severity"), "BDI_Ans": person_data.get("BDI_Ans"), "SuicideIdeationFlag": bool(risk_flags.get("SuicideIdeationFlag", False)), "HighSuicideIdeationFlag": bool(risk_flags.get("HighSuicideIdeationFlag", False))}
        for q in BDI_ITEM_COLS:
            row[q] = safe_number(item_scores.get(q, person_data.get(q)))
        row["CognitiveAffective_Score"] = sum(row[q] for q in BDI_COGNITIVE_AFFECTIVE_COLS if pd.notna(row[q]))
        row["Somatic_Score"] = sum(row[q] for q in BDI_SOMATIC_COLS if pd.notna(row[q]))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)
    df["SexLabel"] = df["Sex"].map(SEX_MAP).fillna("Unknown")
    df["AgeGroup"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=True, include_lowest=True)
    df["BDI_MildPlus_Group"] = np.where(df["BDI_Total"] >= BDI_CUTOFF_MILD, f"BDI >= {BDI_CUTOFF_MILD}", f"BDI < {BDI_CUTOFF_MILD}")
    return df


def build_psqi_df(psqi_data):
    rows = []
    q_cols = ["Q5a_CannotSleep30Min", "Q5b_WakeUpNightOrEarly", "Q5c_Bathroom", "Q5d_Breathe", "Q5e_CoughOrSnore", "Q5f_TooCold", "Q5g_TooHot", "Q5h_BadDream", "Q5i_Pain", "Q6_SleepMedicine", "Q7_DaytimeSleepy", "Q8_EnthusiasmProblem", "Q9_OverallSleepQuality"]
    for person_id, person_data in psqi_data.items():
        derived = person_data.get("Derived", {}) or {}
        comp = person_data.get("PSQI_Component_Scores", {}) or {}
        row = {"ID": person_data.get("ID", person_id), "Sex": person_data.get("Sex"), "Age": person_data.get("Age"), "Valid": bool(person_data.get("Valid", False)), "Invalid_Reasons": "; ".join(person_data.get("Invalid_Reasons", []) or []), "Invalid_Fields": "; ".join(person_data.get("Invalid_Fields", []) or []), "PSQI_Total": safe_number(person_data.get("PSQI_Total")), "PSQI_Ans": person_data.get("PSQI_Ans"), "Q1_BedTime_Hour": person_data.get("Q1_BedTime_Hour"), "Q1_BedTime_Min": person_data.get("Q1_BedTime_Min"), "Q2_SleepLatency_Min": safe_number(person_data.get("Q2_SleepLatency_Min")), "Q3_GetUpTime_Hour": person_data.get("Q3_GetUpTime_Hour"), "Q3_GetUpTime_Min": person_data.get("Q3_GetUpTime_Min"), "Q4_ActualSleep_Hour": person_data.get("Q4_ActualSleep_Hour"), "Q4_ActualSleep_Min": person_data.get("Q4_ActualSleep_Min"), "Q4_ActualSleep_TotalMin": safe_number(person_data.get("Q4_ActualSleep_TotalMin")), "ActualSleepHours": safe_number(derived.get("ActualSleepHours")), "TimeInBedMinutes": safe_number(derived.get("TimeInBedMinutes")), "TimeInBedHours": safe_number(derived.get("TimeInBedHours")), "SleepEfficiencyPercent": safe_number(derived.get("SleepEfficiencyPercent")), "Q2LatencyScore": safe_number(derived.get("Q2LatencyScore")), "C2LatencyRawSum": safe_number(derived.get("C2LatencyRawSum")), "C5DisturbanceRawSum": safe_number(derived.get("C5DisturbanceRawSum")), "C7DaytimeRawSum": safe_number(derived.get("C7DaytimeRawSum"))}
        for q in q_cols:
            row[q] = safe_number(person_data.get(q))
        row["Q5j_Other"] = person_data.get("Q5j_Other")
        for c in PSQI_COMPONENT_COLS:
            row[c] = safe_number(comp.get(c))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)
    df["SexLabel"] = df["Sex"].map(SEX_MAP).fillna("Unknown")
    df["AgeGroup"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=True, include_lowest=True)
    df["PSQI_Group"] = np.where(df["PSQI_Total"] >= PSQI_CUTOFF, f"PSQI >= {PSQI_CUTOFF}", f"PSQI < {PSQI_CUTOFF}")
    return df


def validation_summary(df):
    total = len(df); valid = int(df["Valid"].sum()); invalid = total - valid
    return pd.DataFrame([{"Item": "Valid_Count", "Count": valid, "Percent": count_percent(valid, total)}, {"Item": "Invalid_Count", "Count": invalid, "Percent": count_percent(invalid, total)}, {"Item": "Total_Count", "Count": total, "Percent": "100%" if total else "0%"}])


def numeric_summary(df, cols):
    valid_df = df[df["Valid"]].copy()
    rows = []
    for col in cols:
        if col not in valid_df.columns: continue
        s = pd.to_numeric(valid_df[col], errors="coerce").dropna()
        rows.append({"Variable": col, "N": int(s.count()), "Mean": round(float(s.mean()), 2) if len(s) else np.nan, "Median": round(float(s.median()), 2) if len(s) else np.nan, "Std": round(float(s.std(ddof=1)), 2) if len(s) > 1 else np.nan, "Min": round(float(s.min()), 2) if len(s) else np.nan, "Max": round(float(s.max()), 2) if len(s) else np.nan})
    return pd.DataFrame(rows)


def basic_summary(df, result_col=None, result_order=None):
    valid_df = df[df["Valid"]].copy(); total = len(valid_df); rows = []
    for label in ["Male", "Female", "Unknown"]:
        count = int((valid_df["SexLabel"] == label).sum())
        rows.append({"Section": "Sex", "Item": label, "Count": count, "Percent": count_percent(count, total)})
    age_counts = valid_df["AgeGroup"].value_counts().reindex(AGE_LABELS).fillna(0).astype(int)
    for label, count in age_counts.items():
        rows.append({"Section": "Age", "Item": label, "Count": int(count), "Percent": count_percent(int(count), total)})
    if result_col:
        values = result_order or list(valid_df[result_col].dropna().unique())
        for item in values:
            count = int((valid_df[result_col] == item).sum())
            rows.append({"Section": result_col, "Item": item, "Count": count, "Percent": count_percent(count, total)})
    return pd.DataFrame(rows)


def group_summary(df, group_col, value_col, prefix, extra_cols=None):
    valid_df = df[df["Valid"]].copy(); rows = []
    for group, sub in valid_df.groupby(group_col, observed=False):
        if pd.isna(group) or len(sub) == 0: continue
        row = {group_col: group, "N": len(sub), f"{prefix}_Mean": round(sub[value_col].mean(), 2), f"{prefix}_Median": round(sub[value_col].median(), 2), f"{prefix}_Std": round(sub[value_col].std(ddof=1), 2) if len(sub) > 1 else np.nan}
        if extra_cols:
            for col in extra_cols:
                if col in sub.columns:
                    row[f"{col}_Mean"] = round(sub[col].mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def result_crosstab(df, group_col, result_col, result_order):
    valid_df = df[df["Valid"]].copy()
    count_df = pd.crosstab(valid_df[group_col], valid_df[result_col]).reindex(columns=result_order).fillna(0).astype(int)
    percent_df = count_df.div(count_df.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0) * 100
    return count_df, percent_df.round(1)


def chi_square(count_df):
    from scipy.stats import chi2_contingency
    clean_df = count_df.loc[count_df.sum(axis=1) > 0, count_df.sum(axis=0) > 0]
    if clean_df.shape[0] < 2 or clean_df.shape[1] < 2:
        return pd.DataFrame([{"Chi2": None, "p_value": None, "dof": None, "Significant(p<0.05)": None, "Note": "有效列或欄不足，無法進行 chi-square 檢定"}])
    chi2, p, dof, expected = chi2_contingency(clean_df)
    return pd.DataFrame([{"Chi2": round(chi2, 4), "p_value": round(p, 6), "dof": dof, "Significant(p<0.05)": p < 0.05, "Note": ""}])


def ttest_by_sex(df, value_col):
    from scipy.stats import ttest_ind
    valid_df = df[df["Valid"]].copy()
    male = valid_df[valid_df["SexLabel"] == "Male"][value_col].dropna()
    female = valid_df[valid_df["SexLabel"] == "Female"][value_col].dropna()
    if len(male) < 2 or len(female) < 2:
        return pd.DataFrame([{"Variable": value_col, "t_stat": None, "p_value": None, "Note": "Male 或 Female 樣本數不足"}])
    t_stat, p_value = ttest_ind(male, female, equal_var=False)
    return pd.DataFrame([{"Variable": value_col, "Male_N": len(male), "Female_N": len(female), "Male_Mean": round(male.mean(), 2), "Female_Mean": round(female.mean(), 2), "Mean_Diff_MaleMinusFemale": round(male.mean() - female.mean(), 2), "t_stat": round(t_stat, 4), "p_value": round(p_value, 6), "Significant(p<0.05)": p_value < 0.05, "Note": "Welch t-test"}])


def build_scaled_feature_df(df, feature_cols):
    from sklearn.preprocessing import StandardScaler
    valid_df = df[df["Valid"]].copy().reset_index(drop=True)
    cols = [c for c in feature_cols if c in valid_df.columns]
    feature_df = valid_df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True)).fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df)
    return pd.DataFrame(scaled, columns=feature_df.columns, index=valid_df.index), feature_df


def hierarchical_cluster(df, feature_cols, base_cols, cluster_name_map=None, n_clusters=4):
    from sklearn.cluster import AgglomerativeClustering
    valid_df = df[df["Valid"]].copy().reset_index(drop=True)
    scaled_df, feature_df = build_scaled_feature_df(valid_df, feature_cols)
    n_clusters = min(int(n_clusters), len(valid_df)) if len(valid_df) else 0
    if n_clusters <= 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if n_clusters == 1:
        raw_labels = np.array([1])
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        raw_labels = model.fit_predict(scaled_df) + 1
    cluster_df = valid_df[[c for c in base_cols if c in valid_df.columns]].copy()
    cluster_df["RawCluster"] = raw_labels
    if cluster_name_map:
        cluster_df["Cluster"] = cluster_df["RawCluster"].map(cluster_name_map).fillna(cluster_df["RawCluster"].apply(lambda x: f"Cluster-{int(x)}"))
    else:
        cluster_df["Cluster"] = cluster_df["RawCluster"].apply(lambda x: f"Cluster-{int(x)}")
    profile_df = feature_df.copy(); profile_df["Cluster"] = cluster_df["Cluster"].values
    profile_df = profile_df.groupby("Cluster").mean().round(2)
    profile_df["N"] = cluster_df["Cluster"].value_counts().reindex(profile_df.index).astype(int)
    profile_df = profile_df[["N"] + [c for c in profile_df.columns if c != "N"]]
    raw_profile_df = feature_df.copy(); raw_profile_df["RawCluster"] = cluster_df["RawCluster"].values
    raw_profile_df = raw_profile_df.groupby("RawCluster").mean().round(2)
    raw_profile_df["N"] = cluster_df["RawCluster"].value_counts().reindex(raw_profile_df.index).astype(int)
    raw_profile_df = raw_profile_df[["N"] + [c for c in raw_profile_df.columns if c != "N"]]
    return cluster_df, profile_df, raw_profile_df


def evaluate_cluster_k_range(df, feature_cols, k_min=2, k_max=8):
    """
    比較不同分群數 k 的基本品質指標。

    指標說明：
    - Silhouette 越高越好。
    - Calinski-Harabasz 越高越好。
    - Davies-Bouldin 越低越好。
    - MinClusterSize 太小時，通常表示分群不穩定或切出 outlier-like 小群。
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    valid_df = df[df["Valid"]].copy().reset_index(drop=True)
    if len(valid_df) < 3:
        return pd.DataFrame(columns=[
            "k", "Silhouette", "CalinskiHarabasz", "DaviesBouldin",
            "MinClusterSize", "MaxClusterSize", "ClusterSizes", "Note"
        ])

    scaled_df, _ = build_scaled_feature_df(valid_df, feature_cols)
    x = scaled_df.to_numpy(dtype=float)
    rows = []
    upper_k = min(k_max, len(valid_df) - 1)

    for k in range(k_min, upper_k + 1):
        try:
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
            counts = pd.Series(labels).value_counts().sort_index()
            rows.append({
                "k": k,
                "Silhouette": round(float(silhouette_score(x, labels)), 4),
                "CalinskiHarabasz": round(float(calinski_harabasz_score(x, labels)), 4),
                "DaviesBouldin": round(float(davies_bouldin_score(x, labels)), 4),
                "MinClusterSize": int(counts.min()),
                "MaxClusterSize": int(counts.max()),
                "ClusterSizes": " / ".join(str(int(v)) for v in counts.tolist()),
                "Note": "MinClusterSize < 5，可能是不穩定小群" if int(counts.min()) < 5 else "",
            })
        except Exception as e:
            rows.append({
                "k": k,
                "Silhouette": np.nan,
                "CalinskiHarabasz": np.nan,
                "DaviesBouldin": np.nan,
                "MinClusterSize": np.nan,
                "MaxClusterSize": np.nan,
                "ClusterSizes": "",
                "Note": str(e),
            })

    return pd.DataFrame(rows)


def named_cluster_df(cluster_df, raw_profile_df, raw_info):
    rows = []
    for raw_cluster, info in sorted(raw_info.items(), key=lambda x: x[1]["排序"]):
        ids = cluster_df[cluster_df["RawCluster"] == raw_cluster]["ID"].astype(str).tolist()
        row = {"Cluster": info["Cluster"], "Stage": info["Stage"], "名稱": info["名稱"], "PCA位置": info["PCA位置"], "特徵": info["特徵"], "原始分群": raw_cluster, "人數": len(ids), "ID List": ", ".join(ids)}
        if raw_cluster in raw_profile_df.index:
            for col in raw_profile_df.columns:
                value = raw_profile_df.loc[raw_cluster, col]
                row[col] = round(float(value), 1) if pd.notna(value) and isinstance(value, (int, float, np.integer, np.floating)) else value
        rows.append(row)
    return pd.DataFrame(rows)


def pca_df(df, cluster_df, feature_cols, total_col=None, group_col=None):
    from sklearn.decomposition import PCA
    valid_df = df[df["Valid"]].copy().reset_index(drop=True)
    scaled_df, _ = build_scaled_feature_df(valid_df, feature_cols)
    if len(valid_df) == 0:
        return pd.DataFrame(columns=["ID", "PC1", "PC2", "Cluster", "RawCluster"]), pd.DataFrame({"Component": ["PC1", "PC2"], "ExplainedVarianceRatio": [np.nan, np.nan], "ExplainedVariancePercent": [np.nan, np.nan]})
    if len(valid_df) == 1:
        result = np.array([[0.0, 0.0]])
        ratios = np.array([np.nan, np.nan])
    else:
        pca = PCA(n_components=2)
        result = pca.fit_transform(scaled_df)
        ratios = pca.explained_variance_ratio_
    data = {"ID": valid_df["ID"], "PC1": result[:, 0], "PC2": result[:, 1], "Cluster": cluster_df["Cluster"], "RawCluster": cluster_df["RawCluster"]}
    if total_col and total_col in valid_df.columns: data[total_col] = valid_df[total_col]
    if group_col and group_col in valid_df.columns: data[group_col] = valid_df[group_col]
    explained = pd.DataFrame({"Component": ["PC1", "PC2"], "ExplainedVarianceRatio": ratios, "ExplainedVariancePercent": ratios * 100}).round(4)
    return pd.DataFrame(data), explained


def save_excel_summary(path, sheet_dict):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, sheet_df in sheet_dict.items():
            if isinstance(sheet_df, pd.Series): sheet_df = sheet_df.to_frame()
            sheet_df.to_excel(writer, sheet_name=str(sheet_name)[:31], index=True)


def plot_bar_count_percent(counts, title, xlabel, ylabel, path, colors=None):
    import matplotlib.pyplot as plt
    setup_chinese_font(); total = counts.sum(); ratios = counts / total * 100 if total else counts.astype(float)
    fig, ax = plt.subplots(figsize=(9, 6)); bars = ax.bar(counts.index.astype(str), ratios.values, color=colors)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_ylim(0, max(100, ratios.max() * 1.2 if len(ratios) else 100)); ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, ratio, count in zip(bars, ratios, counts): ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{ratio:.1f}%\n(n={int(count)})", ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def plot_histogram(df, value_col, title, cutoff_list, path):
    import matplotlib.pyplot as plt
    setup_chinese_font(); valid_df = df[df["Valid"]].copy(); s = valid_df[value_col].dropna()
    fig, ax = plt.subplots(figsize=(9, 6)); bins = range(int(s.min()), int(s.max()) + 2) if len(s) else 10
    ax.hist(s, bins=bins, edgecolor="black", alpha=0.8)
    for cutoff, label in cutoff_list: ax.axvline(cutoff, linestyle="--", linewidth=2, label=f"{label} = {cutoff}")
    ax.set_title(title); ax.set_xlabel(value_col); ax.set_ylabel("Count"); ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def plot_mean_bar(summary_df, label_col, mean_col, title, path):
    import matplotlib.pyplot as plt
    setup_chinese_font(); fig, ax = plt.subplots(figsize=(14, 6)); bars = ax.bar(summary_df[label_col], summary_df[mean_col])
    ax.set_title(title); ax.set_xlabel(label_col); ax.set_ylabel(mean_col); ax.tick_params(axis="x", rotation=45); ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, summary_df[mean_col]): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def plot_stacked_percent(percent_df, count_df, title, xlabel, path):
    import matplotlib.pyplot as plt
    setup_chinese_font(); ax = percent_df.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Percent (%)"); ax.set_ylim(0, 110); ax.tick_params(axis="x", rotation=0); ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")
    for i, container in enumerate(ax.containers):
        labels=[]
        for j, value in enumerate(container.datavalues):
            count = int(count_df.iloc[j, i]); labels.append("" if value < 3 or count == 0 else f"{count}\n({value:.1f}%)")
        ax.bar_label(container, labels=labels, label_type="center", fontsize=9, color="black", weight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def plot_correlation_heatmap(corr_df, title, path, annot=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    setup_chinese_font(); size = 18 if len(corr_df) > 12 else 10
    plt.figure(figsize=(size, size)); sns.heatmap(corr_df, annot=annot, cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=0.5, fmt=".2f")
    plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def plot_pca_scatter(pca, explained, title, colors, path):
    import matplotlib.pyplot as plt
    setup_chinese_font(); pc1 = explained.loc[0, "ExplainedVariancePercent"]; pc2 = explained.loc[1, "ExplainedVariancePercent"]
    fig, ax = plt.subplots(figsize=(10, 8))
    for cluster in sorted(pca["Cluster"].dropna().unique()):
        sub = pca[pca["Cluster"] == cluster]
        ax.scatter(sub["PC1"], sub["PC2"], s=60, label=cluster, color=colors.get(cluster, "gray"))
        for _, row in sub.iterrows(): ax.text(row["PC1"], row["PC2"], row["ID"], fontsize=8, alpha=0.75)
    ax.set_title(title); ax.set_xlabel(f"PC1 ({pc1:.1f}%)"); ax.set_ylabel(f"PC2 ({pc2:.1f}%)"); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def plot_dendrogram(df, cluster_df, feature_cols, title, colors, path):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    setup_chinese_font(); valid_df = df[df["Valid"]].copy().reset_index(drop=True); scaled_df, _ = build_scaled_feature_df(valid_df, feature_cols); linked = linkage(scaled_df, method="ward")
    n = len(valid_df); leaf_cluster_map = {i: c for i, c in enumerate(cluster_df["Cluster"].tolist())}; node_leaf_map = {i: [i] for i in range(n)}
    for i, row in enumerate(linked): node_leaf_map[n+i] = node_leaf_map[int(row[0])] + node_leaf_map[int(row[1])]
    def link_color_func(node_id):
        clusters = {leaf_cluster_map[leaf] for leaf in node_leaf_map[int(node_id)]}
        return colors.get(next(iter(clusters)), "gray") if len(clusters) == 1 else "gray"
    fig, ax = plt.subplots(figsize=(18, 8)); dendrogram(linked, labels=valid_df["ID"].tolist(), leaf_rotation=90, leaf_font_size=8, link_color_func=link_color_func, above_threshold_color="gray", ax=ax)
    ax.set_title(title); ax.set_xlabel("ID"); ax.set_ylabel("Distance"); plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def plot_box_by_group(df, group_col, value_col, title, path):
    import matplotlib.pyplot as plt
    setup_chinese_font(); valid_df = df[df["Valid"]].copy(); groups=[]; labels=[]
    for label, sub in valid_df.groupby(group_col, observed=False):
        if pd.isna(label): continue
        values = sub[value_col].dropna()
        if len(values): groups.append(values); labels.append(str(label))
    fig, ax = plt.subplots(figsize=(8, 6)); ax.boxplot(groups, tick_labels=labels, showmeans=True)
    ax.set_title(title); ax.set_xlabel(group_col); ax.set_ylabel(value_col); ax.grid(axis="y", linestyle="--", alpha=0.3); plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def con_cluster_order(): return [info["Cluster"] for _, info in sorted(CON_RAW_CLUSTER_INFO.items(), key=lambda x: x[1]["排序"])]

def con_raw_to_cluster_map(): return {raw: info["Cluster"] for raw, info in CON_RAW_CLUSTER_INFO.items()}


def build_constitution_cluster_distance_df(valid_for_cluster, cluster_df, feature_cols):
    """
    計算每個 ID 到 Constitution 4 個命名 Cluster centroid 的距離。

    距離計算方式：
    - 使用目前 Constitution 分群使用的 feature_cols，也就是 9 個體質分數短名。
    - centroid 以各 RawCluster 在原始分數空間中的平均值計算。
    - dis_1 ~ dis_4 依 CON_RAW_CLUSTER_INFO 的排序對應：
        dis_1 -> Cluster-1
        dis_2 -> Cluster-2
        dis_3 -> Cluster-3A
        dis_4 -> Cluster-3B
    """
    if valid_for_cluster.empty or cluster_df.empty:
        cols = ["ID", "RawCluster", "Cluster", "Stage", "名稱", "dis_1", "dis_2", "dis_3", "dis_4", "MinDistance", "NearestClusterByDistance"]
        return pd.DataFrame(columns=cols)

    feature_matrix = (
        valid_for_cluster[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    feature_matrix = feature_matrix.fillna(feature_matrix.median(numeric_only=True)).fillna(0)

    work_df = cluster_df[["ID", "RawCluster", "Cluster"]].copy().reset_index(drop=True)
    work_df = pd.concat([work_df, feature_matrix.reset_index(drop=True)], axis=1)

    centroid_df = work_df.groupby("RawCluster")[feature_cols].mean()

    ordered_raw_clusters = [
        raw_cluster
        for raw_cluster, _info in sorted(CON_RAW_CLUSTER_INFO.items(), key=lambda x: x[1]["排序"])
    ]

    for idx, raw_cluster in enumerate(ordered_raw_clusters, start=1):
        col = f"dis_{idx}"
        if raw_cluster in centroid_df.index:
            center = centroid_df.loc[raw_cluster, feature_cols].to_numpy(dtype=float)
            values = feature_matrix[feature_cols].to_numpy(dtype=float)
            work_df[col] = np.sqrt(((values - center) ** 2).sum(axis=1)).round(4)
        else:
            work_df[col] = np.nan

    distance_cols = [f"dis_{i}" for i in range(1, 5)]
    work_df["MinDistance"] = work_df[distance_cols].min(axis=1).round(4)

    dis_to_cluster = {
        f"dis_{idx}": CON_RAW_CLUSTER_INFO[raw_cluster]["Cluster"]
        for idx, raw_cluster in enumerate(ordered_raw_clusters, start=1)
    }
    work_df["NearestClusterByDistance"] = work_df[distance_cols].idxmin(axis=1).map(dis_to_cluster)

    info_by_cluster = {info["Cluster"]: info for info in CON_RAW_CLUSTER_INFO.values()}
    work_df["Stage"] = work_df["Cluster"].map(lambda c: info_by_cluster.get(c, {}).get("Stage", ""))
    work_df["名稱"] = work_df["Cluster"].map(lambda c: info_by_cluster.get(c, {}).get("名稱", ""))

    return work_df[["ID", "RawCluster", "Cluster", "Stage", "名稱"] + distance_cols + ["MinDistance", "NearestClusterByDistance"]]


def run_constitution(cons_data):
    out = os.path.join(OUTPUT_DIR, "Constitution"); ensure_dir(out)
    df = build_cons_df(cons_data); valid_df = df[df["Valid"]].copy()
    summary_rows=[]; total=len(valid_df)
    for type_name in CON_TYPE_NAMES:
        counts = valid_df[f"{type_name}_Ans"].value_counts()
        tendency = int(counts.get("基本是", 0)) if type_name == "Type-1（平和體質）" else int(counts.get("傾向是", 0))
        yes = int(counts.get("是", 0)); no = int(counts.get("否", 0)); yt = yes + tendency
        summary_rows.append({"體質": type_name, "是": yes, "是%": count_percent(yes,total), "傾向/基本是": tendency, "傾向/基本是%": count_percent(tendency,total), "否": no, "否%": count_percent(no,total), "是+傾向/基本是": yt, "是+傾向/基本是%": count_percent(yt,total)})
    summary_df = pd.DataFrame(summary_rows)
    binary_df = pd.DataFrame({"ID": valid_df["ID"]})
    for type_name in CON_TYPE_NAMES:
        pos = ["是", "基本是"] if type_name == "Type-1（平和體質）" else ["是", "傾向是"]
        binary_df[type_name] = valid_df[f"{type_name}_Ans"].isin(pos).astype(int).values
    from scipy.stats import ttest_rel
    t_rows=[]
    for i in range(len(CON_TYPE_NAMES)):
        for j in range(i+1,len(CON_TYPE_NAMES)):
            a=binary_df[CON_TYPE_NAMES[i]]; b=binary_df[CON_TYPE_NAMES[j]]; t,p=ttest_rel(a,b)
            t_rows.append({"體質A":CON_TYPE_NAMES[i],"體質B":CON_TYPE_NAMES[j],"A比例%":round(a.mean()*100,0),"B比例%":round(b.mean()*100,0),"差異%":round((a.mean()-b.mean())*100,0),"t_stat":round(t,4),"p_value":round(p,4),"顯著差異(p<0.05)":p<0.05})
    ttest_df=pd.DataFrame(t_rows)
    score_df = pd.DataFrame({short: valid_df[f"{typ}_Score"].values for typ, short in zip(CON_TYPE_NAMES, CON_TYPE_SHORT_NAMES)})
    score_df.insert(0, "ID", valid_df["ID"].values)
    corr_df = score_df.drop(columns=["ID"]).corr(method="pearson").round(3)
    # hierarchical_cluster expects feature cols in df, so add short score cols to valid_df
    valid_for_cluster = valid_df.copy()
    for short in CON_TYPE_SHORT_NAMES: valid_for_cluster[short] = score_df[short].values
    cluster_df, cluster_profile_df, raw_profile_df = hierarchical_cluster(valid_for_cluster, CON_TYPE_SHORT_NAMES, ["ID","SexLabel","Age","AgeGroup"], con_raw_to_cluster_map(), n_clusters=CON_N_CLUSTERS)
    con_distance_df = build_constitution_cluster_distance_df(valid_for_cluster, cluster_df, CON_TYPE_SHORT_NAMES)
    cluster_df = cluster_df.merge(con_distance_df[["ID", "Stage", "名稱", "dis_1", "dis_2", "dis_3", "dis_4", "MinDistance", "NearestClusterByDistance"]], on="ID", how="left")
    named_df = named_cluster_df(cluster_df, raw_profile_df, CON_RAW_CLUSTER_INFO)
    pca, explained = pca_df(valid_for_cluster, cluster_df, CON_TYPE_SHORT_NAMES)
    sex_count, sex_percent = result_crosstab(cluster_df.assign(Valid=True), "SexLabel", "Cluster", con_cluster_order())
    age_count, age_percent = result_crosstab(cluster_df.assign(Valid=True), "AgeGroup", "Cluster", con_cluster_order())
    jobs = [
        ("constitution_score_with_ans.csv", lambda p: df.to_csv(p,index=False,encoding="utf-8-sig")),
        ("constitution_valid_score_with_ans.csv", lambda p: valid_df.to_csv(p,index=False,encoding="utf-8-sig")),
        ("constitution_validation_summary.csv", lambda p: validation_summary(df).to_csv(p,index=False,encoding="utf-8-sig")),
        ("constitution_summary.csv", lambda p: summary_df.to_csv(p,index=False,encoding="utf-8-sig")),
        ("constitution_positive_binary.csv", lambda p: binary_df.to_csv(p,index=False,encoding="utf-8-sig")),
        ("constitution_positive_ttest.csv", lambda p: ttest_df.to_csv(p,index=False,encoding="utf-8-sig")),
        ("constitution_score_matrix.csv", lambda p: score_df.to_csv(p,index=False,encoding="utf-8-sig")),
        ("constitution_correlation.csv", lambda p: corr_df.to_csv(p,encoding="utf-8-sig")),
        ("constitution_hierarchical_cluster.csv", lambda p: cluster_df.to_csv(p,index=False,encoding="utf-8-sig")),
        ("constitution_cluster_profile.csv", lambda p: cluster_profile_df.to_csv(p,encoding="utf-8-sig")),
        ("constitution_raw_cluster_profile.csv", lambda p: raw_profile_df.to_csv(p,encoding="utf-8-sig")),
        ("constitution_named_4_cluster.csv", lambda p: named_df.to_csv(p,index=False,encoding="utf-8-sig")),
        ("Named_4_Cluster.xlsx", lambda p: named_df.to_excel(p,index=False)),
        ("constitution_pca.csv", lambda p: pca.to_csv(p,index=False,encoding="utf-8-sig")),
        ("constitution_pca_explained_variance.csv", lambda p: explained.to_csv(p,index=False,encoding="utf-8-sig")),
        ("sex_cluster_count.csv", lambda p: sex_count.to_csv(p,encoding="utf-8-sig")),
        ("sex_cluster_percent.csv", lambda p: sex_percent.to_csv(p,encoding="utf-8-sig")),
        ("sex_cluster_chisquare.csv", lambda p: chi_square(sex_count).to_csv(p,index=False,encoding="utf-8-sig")),
        ("age_cluster_count.csv", lambda p: age_count.to_csv(p,encoding="utf-8-sig")),
        ("age_cluster_percent.csv", lambda p: age_percent.to_csv(p,encoding="utf-8-sig")),
        ("age_cluster_chisquare.csv", lambda p: chi_square(age_count).to_csv(p,index=False,encoding="utf-8-sig")),
        ("Constitution_Analysis_Summary.xlsx", lambda p: save_excel_summary(p, {"validation":validation_summary(df),"summary":summary_df,"ttest":ttest_df,"correlation":corr_df,"cluster":cluster_df,"cluster_profile":cluster_profile_df,"named_cluster":named_df,"pca_explained":explained,"sex_cluster_count":sex_count,"age_cluster_count":age_count})),
        ("constitution_correlation_heatmap.png", lambda p: plot_correlation_heatmap(corr_df,"體質相關性 Heatmap",p,annot=True)),
        ("constitution_pca_scatter.png", lambda p: plot_pca_scatter(pca,explained,"PCA 體質分布圖",CON_CLUSTER_COLORS,p)),
        ("constitution_hierarchical_dendrogram.png", lambda p: plot_dendrogram(valid_for_cluster,cluster_df,CON_TYPE_SHORT_NAMES,"體質階層式分群 Dendrogram",CON_CLUSTER_COLORS,p)),
        ("cluster_population_ratio_barchart.png", lambda p: plot_bar_count_percent(cluster_df["Cluster"].value_counts().reindex(con_cluster_order()).fillna(0).astype(int),"各 Cluster 群人數比例圖","Cluster","Percent (%)",p,[CON_CLUSTER_COLORS.get(c,"gray") for c in con_cluster_order()])),
        ("sex_cluster_distribution.png", lambda p: plot_stacked_percent(sex_percent,sex_count,"Sex × Cluster 分布圖","Sex",p)),
        ("age_cluster_distribution.png", lambda p: plot_stacked_percent(age_percent,age_count,"Age × Cluster 分布圖","Age Group",p)),
    ]
    for file, func in jobs: save_if_needed(os.path.join(out,file), func, file)
    return {"df": df, "valid_df": valid_df, "cluster_df": cluster_df, "pca_df": pca, "distance_df": con_distance_df}


def analyze_questionnaire(prefix, df, feature_cols, item_cols, item_labels, total_col, group_col, result_order, out, histogram_cutoffs, n_clusters=2, cluster_feature_cols=None):
    ensure_dir(out); valid_df = df[df["Valid"]].copy()
    numeric_cols = ["Age", total_col] + [c for c in ["CognitiveAffective_Score","Somatic_Score","Q2_SleepLatency_Min","ActualSleepHours","TimeInBedHours","SleepEfficiencyPercent","C2LatencyRawSum","C5DisturbanceRawSum","C7DaytimeRawSum"] if c in df.columns] + item_cols
    item_summary_rows=[]
    for c in item_cols:
        s=valid_df[c].dropna(); item_summary_rows.append({"Item":c,"Label":item_labels.get(c,c),"Mean":round(s.mean(),2),"Median":round(s.median(),2),"Std":round(s.std(ddof=1),2),"Min":int(s.min()) if len(s) else np.nan,"Max":int(s.max()) if len(s) else np.nan})
    item_summary_df=pd.DataFrame(item_summary_rows)
    dist_rows=[]; total=len(valid_df)
    for c in item_cols:
        counts=valid_df[c].value_counts().reindex([0,1,2,3]).fillna(0).astype(int)
        for score,count in counts.items(): dist_rows.append({"Item":c,"Label":item_labels.get(c,c),"Score":int(score),"Count":int(count),"Percent":round(count/total*100,1) if total else 0})
    item_dist_df=pd.DataFrame(dist_rows)
    sex_summary=group_summary(df,"SexLabel",total_col,prefix)
    age_summary=group_summary(df,"AgeGroup",total_col,prefix)
    sex_count, sex_percent = result_crosstab(df,"SexLabel",group_col,result_order)
    age_count, age_percent = result_crosstab(df,"AgeGroup",group_col,result_order)
    corr_df = df[df["Valid"]][[c for c in feature_cols if c in df.columns]].corr(method="pearson").round(3)
    cluster_feature_cols = cluster_feature_cols or feature_cols
    cluster_eval_df = evaluate_cluster_k_range(df, cluster_feature_cols)
    cluster_df, cluster_profile_df, raw_profile_df = hierarchical_cluster(
        df,
        cluster_feature_cols,
        ["ID", "SexLabel", "Age", "AgeGroup", total_col, group_col],
        n_clusters=n_clusters,
    )
    raw_info = {
        i: {
            "排序": i,
            "Cluster": f"Cluster-{i}",
            "Stage": f"Stage-{i}",
            "名稱": f"{prefix} Cluster-{i}",
            "PCA位置": "待依 PCA 圖判讀",
            "特徵": "待依 cluster profile 判讀",
        }
        for i in range(1, n_clusters + 1)
    }
    named_df = named_cluster_df(cluster_df, raw_profile_df, raw_info)
    pca, explained = pca_df(df, cluster_df, cluster_feature_cols, total_col, group_col)
    jobs = [
        (f"{prefix.lower()}_score_with_ans.csv", lambda p: df.to_csv(p,index=False,encoding="utf-8-sig")),
        (f"{prefix.lower()}_valid_score_with_ans.csv", lambda p: valid_df.to_csv(p,index=False,encoding="utf-8-sig")),
        (f"{prefix.lower()}_validation_summary.csv", lambda p: validation_summary(df).to_csv(p,index=False,encoding="utf-8-sig")),
        (f"{prefix.lower()}_basic_summary.csv", lambda p: basic_summary(df, group_col, result_order).to_csv(p,index=False,encoding="utf-8-sig")),
        (f"{prefix.lower()}_numeric_summary.csv", lambda p: numeric_summary(df,numeric_cols).to_csv(p,index=False,encoding="utf-8-sig")),
        (f"{prefix.lower()}_item_summary.csv", lambda p: item_summary_df.to_csv(p,index=False,encoding="utf-8-sig")),
        (f"{prefix.lower()}_item_score_distribution.csv", lambda p: item_dist_df.to_csv(p,index=False,encoding="utf-8-sig")),
        (f"sex_{prefix.lower()}_summary.csv", lambda p: sex_summary.to_csv(p,index=False,encoding="utf-8-sig")),
        (f"age_{prefix.lower()}_summary.csv", lambda p: age_summary.to_csv(p,index=False,encoding="utf-8-sig")),
        (f"sex_{prefix.lower()}_count.csv", lambda p: sex_count.to_csv(p,encoding="utf-8-sig")),
        (f"sex_{prefix.lower()}_percent.csv", lambda p: sex_percent.to_csv(p,encoding="utf-8-sig")),
        (f"sex_{prefix.lower()}_chisquare.csv", lambda p: chi_square(sex_count).to_csv(p,index=False,encoding="utf-8-sig")),
        (f"sex_{prefix.lower()}_ttest.csv", lambda p: ttest_by_sex(df,total_col).to_csv(p,index=False,encoding="utf-8-sig")),
        (f"age_{prefix.lower()}_count.csv", lambda p: age_count.to_csv(p,encoding="utf-8-sig")),
        (f"age_{prefix.lower()}_percent.csv", lambda p: age_percent.to_csv(p,encoding="utf-8-sig")),
        (f"age_{prefix.lower()}_chisquare.csv", lambda p: chi_square(age_count).to_csv(p,index=False,encoding="utf-8-sig")),
        (f"{prefix.lower()}_correlation.csv", lambda p: corr_df.to_csv(p,encoding="utf-8-sig")),
        (f"{prefix.lower()}_hierarchical_cluster.csv", lambda p: cluster_df.to_csv(p,index=False,encoding="utf-8-sig")),
        (f"{prefix.lower()}_cluster_k_evaluation.csv", lambda p: cluster_eval_df.to_csv(p,index=False,encoding="utf-8-sig")),
        (f"{prefix.lower()}_cluster_profile.csv", lambda p: cluster_profile_df.to_csv(p,encoding="utf-8-sig")),
        (f"{prefix.lower()}_raw_cluster_profile.csv", lambda p: raw_profile_df.to_csv(p,encoding="utf-8-sig")),
        (f"{prefix.lower()}_named_{n_clusters}_cluster.csv", lambda p: named_df.to_csv(p,index=False,encoding="utf-8-sig")),
        (f"Named_{n_clusters}_Cluster.xlsx", lambda p: named_df.to_excel(p,index=False)),
        (f"{prefix.lower()}_pca.csv", lambda p: pca.to_csv(p,index=False,encoding="utf-8-sig")),
        (f"{prefix.lower()}_pca_explained_variance.csv", lambda p: explained.to_csv(p,index=False,encoding="utf-8-sig")),
        (f"{prefix}_Analysis_Summary.xlsx", lambda p: save_excel_summary(p,{"validation":validation_summary(df),"basic_summary":basic_summary(df,group_col,result_order),"numeric_summary":numeric_summary(df,numeric_cols),"item_summary":item_summary_df,"item_distribution":item_dist_df,"sex_summary":sex_summary,"age_summary":age_summary,"sex_count":sex_count,"sex_percent":sex_percent,"sex_chisquare":chi_square(sex_count),"sex_ttest":ttest_by_sex(df,total_col),"age_count":age_count,"age_percent":age_percent,"age_chisquare":chi_square(age_count),"correlation":corr_df,"cluster_k_evaluation":cluster_eval_df,"cluster_profile":cluster_profile_df,"raw_cluster_profile":raw_profile_df,"named_cluster":named_df,"pca_explained":explained})),
        (f"{prefix.lower()}_result_ratio_barchart.png", lambda p: plot_bar_count_percent(valid_df[group_col].value_counts().reindex(result_order).fillna(0).astype(int),f"{prefix} Result 比例圖",group_col,"Percent (%)",p)),
        ("sex_ratio_barchart.png", lambda p: plot_bar_count_percent(valid_df["SexLabel"].value_counts().reindex(["Male","Female","Unknown"]).fillna(0).astype(int),"Sex 比例圖","Sex","Percent (%)",p)),
        ("age_ratio_barchart.png", lambda p: plot_bar_count_percent(valid_df["AgeGroup"].value_counts().reindex(AGE_LABELS).fillna(0).astype(int),"Age Distribution 比例圖","Age Group","Percent (%)",p)),
        (f"{prefix.lower()}_total_score_histogram.png", lambda p: plot_histogram(df,total_col,f"{prefix} Total Score Distribution",histogram_cutoffs,p)),
        (f"{prefix.lower()}_item_mean_barchart.png", lambda p: plot_mean_bar(item_summary_df,"Label","Mean",f"{prefix} Item/Component Mean",p)),
        (f"{prefix.lower()}_total_by_sex_boxplot.png", lambda p: plot_box_by_group(df,"SexLabel",total_col,f"{prefix} Total by Sex",p)),
        (f"{prefix.lower()}_total_by_age_boxplot.png", lambda p: plot_box_by_group(df,"AgeGroup",total_col,f"{prefix} Total by Age Group",p)),
        (f"sex_{prefix.lower()}_distribution.png", lambda p: plot_stacked_percent(sex_percent,sex_count,f"Sex × {prefix} Result 分布圖","Sex",p)),
        (f"age_{prefix.lower()}_distribution.png", lambda p: plot_stacked_percent(age_percent,age_count,f"Age × {prefix} Result 分布圖","Age Group",p)),
        (f"{prefix.lower()}_correlation_heatmap.png", lambda p: plot_correlation_heatmap(corr_df,f"{prefix} Variables Correlation Heatmap",p,annot=len(corr_df)<=12)),
        (f"{prefix.lower()}_cluster_population_ratio_barchart.png", lambda p: plot_bar_count_percent(cluster_df["Cluster"].value_counts().reindex([f"Cluster-{i}" for i in range(1,n_clusters+1)]).fillna(0).astype(int),f"{prefix} Cluster 人數比例圖","Cluster","Percent (%)",p,[GENERIC_CLUSTER_COLORS.get(f"Cluster-{i}","gray") for i in range(1,n_clusters+1)])),
        (f"{prefix.lower()}_pca_scatter.png", lambda p: plot_pca_scatter(pca,explained,f"PCA {prefix} 分布圖",GENERIC_CLUSTER_COLORS,p)),
        (f"{prefix.lower()}_hierarchical_dendrogram.png", lambda p: plot_dendrogram(df,cluster_df,feature_cols,f"{prefix} 階層式分群 Dendrogram",GENERIC_CLUSTER_COLORS,p)),
    ]
    for file, func in jobs: save_if_needed(os.path.join(out,file), func, file)
    return {"df": df, "valid_df": valid_df, "cluster_df": cluster_df, "cluster_eval_df": cluster_eval_df, "pca_df": pca}


def run_bdi(bdi_data):
    df = build_bdi_df(bdi_data)
    return analyze_questionnaire("BDI", df, BDI_FEATURE_COLS, BDI_ITEM_COLS, BDI_ITEM_LABELS, "BDI_Total", "BDI_MildPlus_Group", [f"BDI < {BDI_CUTOFF_MILD}", f"BDI >= {BDI_CUTOFF_MILD}"], os.path.join(OUTPUT_DIR,"BDI"), [(14,"Mild"),(20,"Moderate"),(29,"Severe")], n_clusters=BDI_N_CLUSTERS, cluster_feature_cols=BDI_FEATURE_COLS)


def run_psqi(psqi_data):
    df = build_psqi_df(psqi_data)
    return analyze_questionnaire("PSQI", df, PSQI_FEATURE_COLS, PSQI_COMPONENT_COLS, PSQI_COMPONENT_LABELS, "PSQI_Total", "PSQI_Group", [f"PSQI < {PSQI_CUTOFF}", f"PSQI >= {PSQI_CUTOFF}"], os.path.join(OUTPUT_DIR,"PSQI"), [(PSQI_CUTOFF,"Cutoff")], n_clusters=PSQI_N_CLUSTERS, cluster_feature_cols=PSQI_CLUSTER_FEATURE_COLS)



def format_distance_00(value):
    """把距離格式化成 00.00，例如 03.52。"""
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):05.2f}"
    except Exception:
        return ""



def _format_valid_flag(value):
    return "Valid" if bool(value) else "Invalid"


def _string_id_list(values):
    return ", ".join(sorted(pd.Series(values).dropna().astype(str).tolist()))


def build_generic_cluster_distance_df(valid_df, cluster_df, feature_cols, cluster_prefix="Cluster"):
    """
    計算 BDI / PSQI 每個 valid ID 到各 cluster centroid 的距離。

    - feature_cols 使用該量表實際分群的欄位。
    - dis_1、dis_2 ... 對應 Cluster-1、Cluster-2 ...。
    - Distance 為該 ID 所屬 cluster 的 centroid distance。
    """
    base_cols = ["ID", "RawCluster", "Cluster"]
    if valid_df is None or valid_df.empty or cluster_df is None or cluster_df.empty:
        return pd.DataFrame(columns=base_cols + ["Distance"])

    valid_work = valid_df.copy().reset_index(drop=True)
    cluster_work = cluster_df[[c for c in base_cols if c in cluster_df.columns]].copy().reset_index(drop=True)
    cols = [c for c in feature_cols if c in valid_work.columns]
    if not cols:
        cluster_work["Distance"] = np.nan
        return cluster_work

    feature_matrix = (
        valid_work[cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    feature_matrix = feature_matrix.fillna(feature_matrix.median(numeric_only=True)).fillna(0)

    work_df = pd.concat([cluster_work, feature_matrix.reset_index(drop=True)], axis=1)
    centroid_df = work_df.groupby("RawCluster")[cols].mean()
    ordered_raw_clusters = sorted(pd.Series(cluster_work["RawCluster"]).dropna().astype(int).unique().tolist())

    values = feature_matrix[cols].to_numpy(dtype=float)
    for idx, raw_cluster in enumerate(ordered_raw_clusters, start=1):
        col = f"dis_{idx}"
        if raw_cluster in centroid_df.index:
            center = centroid_df.loc[raw_cluster, cols].to_numpy(dtype=float)
            work_df[col] = np.sqrt(((values - center) ** 2).sum(axis=1)).round(4)
        else:
            work_df[col] = np.nan

    def own_distance(row):
        try:
            raw = int(row["RawCluster"])
            pos = ordered_raw_clusters.index(raw) + 1
            return row.get(f"dis_{pos}", np.nan)
        except Exception:
            return np.nan

    work_df["Distance"] = work_df.apply(own_distance, axis=1).round(4)
    return work_df[[c for c in base_cols + [f"dis_{i}" for i in range(1, len(ordered_raw_clusters) + 1)] + ["Distance"] if c in work_df.columns]]


def build_scale_cluster_table(result, scale_name, total_col=None, ans_col=None, distance_df=None):
    """
    建立單一量表的 all-ID cluster table。

    Invalid 的 ID 會保留，cluster 欄位填入 Invalid。
    Valid 但沒有分群資料時，cluster 欄位填入 MissingCluster，方便除錯。
    """
    df = result["df"].copy()
    keep_cols = ["ID", "Valid", "Invalid_Reasons", "Invalid_Fields"]
    for col in ["Sex", "Age", "SexLabel", total_col, ans_col]:
        if col and col in df.columns and col not in keep_cols:
            keep_cols.append(col)
    table = df[keep_cols].copy()
    table = table.rename(columns={
        "Valid": f"{scale_name}_Valid",
        "Invalid_Reasons": f"{scale_name}_Invalid_Reasons",
        "Invalid_Fields": f"{scale_name}_Invalid_Fields",
    })

    cluster_source = distance_df if distance_df is not None and not distance_df.empty else result.get("cluster_df", pd.DataFrame())
    cluster_cols = ["ID", "RawCluster", "Cluster", "Distance"] + [c for c in cluster_source.columns if c.startswith("dis_")]
    cluster_cols = [c for c in cluster_cols if c in cluster_source.columns]
    if cluster_cols:
        cluster_part = cluster_source[cluster_cols].copy()
        rename_map = {c: f"{scale_name}_{c}" for c in cluster_part.columns if c != "ID"}
        cluster_part = cluster_part.rename(columns=rename_map)
        table = table.merge(cluster_part, on="ID", how="left")

    cluster_col = f"{scale_name}_Cluster"
    raw_col = f"{scale_name}_RawCluster"
    distance_col = f"{scale_name}_Distance"
    valid_col = f"{scale_name}_Valid"

    if cluster_col not in table.columns:
        table[cluster_col] = np.nan
    if raw_col not in table.columns:
        table[raw_col] = np.nan
    if distance_col not in table.columns:
        table[distance_col] = np.nan

    table[cluster_col] = np.where(
        table[valid_col].astype(bool),
        table[cluster_col].fillna("MissingCluster"),
        "Invalid",
    )
    table[raw_col] = np.where(table[valid_col].astype(bool), table[raw_col], "Invalid")

    for col in [c for c in table.columns if c.startswith(f"{scale_name}_dis_") or c == distance_col]:
        table[col] = table[col].map(format_distance_00)

    table[f"{scale_name}_Status"] = table[valid_col].map(_format_valid_flag)
    return table


def build_integrated_cluster_outputs(all_ids_df, scale_tables):
    """
    建立 Integrated 的三個主要輸出：

    1. integrated_common_valid_ids.csv
       保留全部 ID。每個量表各自標示 Valid / Invalid、Cluster、RawCluster、Distance。

    2. integrated_ids_cluster.csv
       每個 ID 一列，列出 Cons / BDI / PSQI 的 cluster type；Invalid 顯示 Invalid。

    3. integrated_cluster_ids.csv
       每個 Scale × Cluster 一列，列出 ID list 與 distance min / mean / max。
    """
    common = all_ids_df.copy()
    for scale_name, table in scale_tables.items():
        common = common.merge(table, on="ID", how="left")

    common = common.sort_values("ID").reset_index(drop=True)

    ids_cols = ["ID"]
    preferred_cols_by_scale = {
        "Cons": ["Cons_Status", "Cons_Cluster", "Cons_RawCluster", "Cons_Distance", "Cons_dis_1", "Cons_dis_2", "Cons_dis_3", "Cons_dis_4"],
        "BDI": ["BDI_Status", "BDI_Cluster", "BDI_RawCluster", "BDI_Distance", "BDI_dis_1", "BDI_dis_2"],
        "PSQI": ["PSQI_Status", "PSQI_Cluster", "PSQI_RawCluster", "PSQI_Distance", "PSQI_dis_1", "PSQI_dis_2"],
    }
    for _scale, cols in preferred_cols_by_scale.items():
        ids_cols.extend([c for c in cols if c in common.columns])
    ids_cluster_df = common[[c for c in ids_cols if c in common.columns]].copy()

    cluster_rows = []
    for scale_name, table in scale_tables.items():
        cluster_col = f"{scale_name}_Cluster"
        distance_col = f"{scale_name}_Distance"
        status_col = f"{scale_name}_Status"
        for cluster_name, sub in table.groupby(cluster_col, dropna=False):
            cluster_name = "Invalid" if pd.isna(cluster_name) else str(cluster_name)
            distances = pd.to_numeric(sub.get(distance_col, pd.Series(dtype=float)).replace("", np.nan), errors="coerce").dropna()
            id_list = _string_id_list(sub["ID"])
            cluster_rows.append({
                "Scale": scale_name,
                "Cluster": cluster_name,
                "Status": "Invalid" if cluster_name == "Invalid" else "Valid",
                "Count": int(len(sub)),
                "ID_List": id_list,
                "Min_Distance": format_distance_00(distances.min()) if len(distances) else "",
                "Mean_Distance": format_distance_00(distances.mean()) if len(distances) else "",
                "Max_Distance": format_distance_00(distances.max()) if len(distances) else "",
            })

    cluster_ids_df = pd.DataFrame(cluster_rows)
    if not cluster_ids_df.empty:
        scale_order = {"Cons": 1, "BDI": 2, "PSQI": 3}
        cluster_ids_df["_scale_sort"] = cluster_ids_df["Scale"].map(scale_order).fillna(99)
        cluster_ids_df["_invalid_sort"] = (cluster_ids_df["Cluster"] == "Invalid").astype(int)
        cluster_ids_df = (
            cluster_ids_df
            .sort_values(["_scale_sort", "_invalid_sort", "Cluster"])
            .drop(columns=["_scale_sort", "_invalid_sort"])
            .reset_index(drop=True)
        )

    return common, ids_cluster_df, cluster_ids_df


def run_integrated_outputs(cons_result, bdi_result, psqi_result):
    out = os.path.join(OUTPUT_DIR, "Integrated"); ensure_dir(out)
    dfs = []
    for name, result, total_col in [("Constitution", cons_result, None), ("BDI", bdi_result, "BDI_Total"), ("PSQI", psqi_result, "PSQI_Total")]:
        df = result["df"]
        row = {"Scale": name, "Total_Count": len(df), "Valid_Count": int(df["Valid"].sum()), "Invalid_Count": int((~df["Valid"]).sum()), "Valid_Percent": count_percent(int(df["Valid"].sum()), len(df), 1)}
        if total_col and total_col in df.columns:
            v = df[df["Valid"]][total_col].dropna(); row.update({"Score_Mean": round(v.mean(),2), "Score_Median": round(v.median(),2), "Score_Std": round(v.std(ddof=1),2)})
        dfs.append(row)
    summary = pd.DataFrame(dfs)

    all_ids = sorted(
        set(cons_result["df"]["ID"].astype(str)) |
        set(bdi_result["df"]["ID"].astype(str)) |
        set(psqi_result["df"]["ID"].astype(str))
    )
    all_ids_df = pd.DataFrame({"ID": all_ids})

    # 確保三個 result 的 ID 都用 str 對齊，避免 001 與 1 這類格式問題。
    for result in [cons_result, bdi_result, psqi_result]:
        for key in ["df", "valid_df", "cluster_df"]:
            if key in result and isinstance(result[key], pd.DataFrame) and "ID" in result[key].columns:
                result[key] = result[key].copy()
                result[key]["ID"] = result[key]["ID"].astype(str)
        if "distance_df" in result and isinstance(result["distance_df"], pd.DataFrame) and "ID" in result["distance_df"].columns:
            result["distance_df"] = result["distance_df"].copy()
            result["distance_df"]["ID"] = result["distance_df"]["ID"].astype(str)

    bdi_distance_df = build_generic_cluster_distance_df(
        bdi_result["valid_df"],
        bdi_result["cluster_df"],
        BDI_FEATURE_COLS,
        cluster_prefix="BDI",
    )
    psqi_distance_df = build_generic_cluster_distance_df(
        psqi_result["valid_df"],
        psqi_result["cluster_df"],
        PSQI_CLUSTER_FEATURE_COLS,
        cluster_prefix="PSQI",
    )

    scale_tables = OrderedDict([
        ("Cons", build_scale_cluster_table(cons_result, "Cons", distance_df=cons_result.get("distance_df", pd.DataFrame()))),
        ("BDI", build_scale_cluster_table(bdi_result, "BDI", total_col="BDI_Total", ans_col="BDI_Ans", distance_df=bdi_distance_df)),
        ("PSQI", build_scale_cluster_table(psqi_result, "PSQI", total_col="PSQI_Total", ans_col="PSQI_Ans", distance_df=psqi_distance_df)),
    ])

    common, ids_cluster_df, cluster_ids_df = build_integrated_cluster_outputs(all_ids_df, scale_tables)

    save_if_needed(os.path.join(out,"integrated_scale_summary.csv"), lambda p: summary.to_csv(p,index=False,encoding="utf-8-sig"), "integrated_scale_summary.csv")
    save_if_needed(os.path.join(out,"integrated_common_valid_ids.csv"), lambda p: common.to_csv(p,index=False,encoding="utf-8-sig"), "integrated_common_valid_ids.csv")
    save_if_needed(os.path.join(out,"integrated_ids_cluster.csv"), lambda p: ids_cluster_df.to_csv(p,index=False,encoding="utf-8-sig"), "integrated_ids_cluster.csv")
    save_if_needed(os.path.join(out,"integrated_cluster_ids.csv"), lambda p: cluster_ids_df.to_csv(p,index=False,encoding="utf-8-sig"), "integrated_cluster_ids.csv")
    save_if_needed(
        os.path.join(out,"Integrated_Statics_Summary.xlsx"),
        lambda p: save_excel_summary(p,{
            "scale_summary": summary,
            "common_valid_ids": common,
            "ids_cluster": ids_cluster_df,
            "cluster_ids": cluster_ids_df,
            "cons_id_cluster": scale_tables["Cons"],
            "bdi_id_cluster": scale_tables["BDI"],
            "psqi_id_cluster": scale_tables["PSQI"],
        }),
        "Integrated_Statics_Summary.xlsx",
    )

def write_analysis_statics_md():
    text = """# AnalysisStatics.md

## 程式

`AnalysisStatics.py` 會整合 Constitution、BDI-II、PSQI 三個 Statistics 分析流程。

本程式直接讀取 Basic 階段輸出的 `RecordOriginal_3_sets.pkl`。
（不再重新讀取三份 original Excel，也不再依賴 `_temp_BDI_value_*.json` 或 `_temp_PSQI_value_*.json`）

## 執行方式

```bash
python AnalysisStatics.py
```

預設輸入檔案必須與程式放在同一個目錄：

```text
RecordOriginal_3_sets.pkl
```

預設輸出目錄：

```text
AnalysisStatics_Data/
```

## 輸入 pkl 格式

`RecordOriginal_3_sets.pkl` 必須是一個 dict / OrderedDict，包含三個 key：

```python
valid_data = {
    "Constitution": con_score_dict,
    "BDI": bdi_score_dict,
    "PSQI": psqi_score_dict,
}
```

### Constitution record

每個 ID 對應一筆 dict：

```python
{
    "ID": "A001",
    "Sex": 1,
    "Age": 18,
    "Valid": True,
    "Invalid_Reasons": [],
    "Type-1（平和體質）": {"Q1": 3, ..., "Score": 66, "Ans": "否"},
    ...,
    "Type-9（特稟體質）": {"Q1": 2, ..., "Score": 25, "Ans": "否"},
}
```

Statistics 會使用每個 Type 的 `Score` 與 `Ans`。

### BDI record

每個 ID 對應一筆 dict：

```python
{
    "ID": "A001",
    "Sex": 1,
    "Age": 18,
    "Valid": True,
    "Invalid_Reasons": [],
    "BDI_Item_Scores": {"Q1_Sadness": 0, ..., "Q21_LossOfSexInterest": 0},
    "BDI_Total": 5,
    "BDI_Severity": "Minimal",
    "BDI_Ans": "極輕微/無明顯憂鬱",
    "Risk_Flags": {"SuicideIdeationFlag": False, "HighSuicideIdeationFlag": False},
}
```

Statistics 會使用 `BDI_Item_Scores`、`BDI_Total`、`BDI_Ans` 與 `Risk_Flags`。

### PSQI record

每個 ID 對應一筆 dict：

```python
{
    "ID": "A001",
    "Sex": 1,
    "Age": 18,
    "Valid": True,
    "Invalid_Reasons": [],
    "Q2_SleepLatency_Min": 30,
    "Q4_ActualSleep_TotalMin": 360,
    "Derived": {
        "ActualSleepHours": 6.0,
        "TimeInBedMinutes": 420,
        "TimeInBedHours": 7.0,
        "SleepEfficiencyPercent": 85.71,
        "Q2LatencyScore": 1,
        "C2LatencyRawSum": 4,
        "C5DisturbanceRawSum": 7,
        "C7DaytimeRawSum": 2,
    },
    "PSQI_Component_Scores": {
        "C1_SleepQuality": 1,
        ...,
        "C7_DaytimeDysfunction": 1,
    },
    "PSQI_Total": 6,
    "PSQI_Ans": "睡眠品質障礙",
}
```

Statistics 會使用 `PSQI_Total`、`PSQI_Component_Scores`、`Derived` 與主要原始欄位。

## 輸出目錄安排

```text
AnalysisStatics_Data/
├── Constitution/
├── BDI/
├── PSQI/
├── Integrated/
└── AnalysisStatics.md
```

## Constitution 輸出格式

| 檔案 | 說明 |
|---|---|
| `constitution_score_with_ans.csv` | 全部 Constitution record 展平成表格。 |
| `constitution_valid_score_with_ans.csv` | 只保留 Valid=True。 |
| `constitution_validation_summary.csv` | Valid / Invalid / Total 數量。 |
| `constitution_summary.csv` | 9 種體質的「是 / 傾向或基本是 / 否」人數與百分比。 |
| `constitution_positive_binary.csv` | 每個 ID × 9 種體質的陽性二元表。 |
| `constitution_positive_ttest.csv` | 9 種體質陽性比例的 paired t-test。 |
| `constitution_score_matrix.csv` | 9 種體質 Score 矩陣。 |
| `constitution_correlation.csv` | 9 種體質 Score Pearson correlation。 |
| `constitution_hierarchical_cluster.csv` | 階層式分群結果，含 RawCluster 與命名後 Cluster。 |
| `constitution_cluster_profile.csv` | 各 Cluster 的體質平均分數。 |
| `constitution_raw_cluster_profile.csv` | 以 RawCluster 為 index 的 profile。 |
| `constitution_named_4_cluster.csv` / `Named_4_Cluster.xlsx` | 4 群命名總表，含 Stage、名稱、PCA位置、特徵、ID List。 |
| `constitution_pca.csv` | PCA 2D 座標。 |
| `constitution_pca_explained_variance.csv` | PC1/PC2 explained variance。 |
| `sex_cluster_count.csv`, `sex_cluster_percent.csv`, `sex_cluster_chisquare.csv` | Sex × Cluster 統計。 |
| `age_cluster_count.csv`, `age_cluster_percent.csv`, `age_cluster_chisquare.csv` | AgeGroup × Cluster 統計。 |
| `Constitution_Analysis_Summary.xlsx` | 主要表格整合成 Excel sheets。 |
| `*.png` | heatmap、PCA、dendrogram、Cluster 比例、Sex/Age stacked bar。 |

## BDI 輸出格式

| 檔案 | 說明 |
|---|---|
| `bdi_score_with_ans.csv` | 全部 BDI record 展平成表格。 |
| `bdi_valid_score_with_ans.csv` | 只保留 Valid=True。 |
| `bdi_validation_summary.csv` | Valid / Invalid / Total 數量。 |
| `bdi_basic_summary.csv` | Sex、AgeGroup、BDI Mild+ 統計。 |
| `bdi_numeric_summary.csv` | Age、BDI_Total、分項與症狀群的平均/中位數/標準差。 |
| `bdi_item_summary.csv` | Q1~Q21 每題平均與範圍。 |
| `bdi_item_score_distribution.csv` | Q1~Q21 的 0/1/2/3 分分布。 |
| `sex_bdi_*`, `age_bdi_*` | Sex/AgeGroup 與 BDI Mild+ 的 count、percent、chi-square、t-test。 |
| `bdi_correlation.csv` | BDI_Total、症狀群、Q1~Q21 correlation。 |
| `bdi_hierarchical_cluster.csv` | BDI 階層式分群結果。 |
| `bdi_cluster_k_evaluation.csv` | BDI 使用 k=2~8 的 Silhouette、Calinski-Harabasz、Davies-Bouldin、群大小比較。 |
| `bdi_cluster_profile.csv`, `bdi_raw_cluster_profile.csv` | Cluster feature profile。 |
| `bdi_named_2_cluster.csv` / `Named_2_Cluster.xlsx` | BDI 2 群命名總表。 |
| `bdi_pca.csv`, `bdi_pca_explained_variance.csv` | PCA 座標與解釋變異。 |
| `BDI_Analysis_Summary.xlsx` | 主要表格整合成 Excel sheets。 |
| `*.png` | severity/result ratio、histogram、boxplot、heatmap、PCA、dendrogram、cluster ratio。 |

## PSQI 輸出格式

| 檔案 | 說明 |
|---|---|
| `psqi_score_with_ans.csv` | 全部 PSQI record 展平成表格。 |
| `psqi_valid_score_with_ans.csv` | 只保留 Valid=True。 |
| `psqi_validation_summary.csv` | Valid / Invalid / Total 數量。 |
| `psqi_basic_summary.csv` | Sex、AgeGroup、PSQI cutoff 統計。 |
| `psqi_numeric_summary.csv` | Age、PSQI_Total、睡眠時間、效率、component scores 的數值統計。 |
| `psqi_item_summary.csv` | C1~C7 component 平均與範圍。 |
| `psqi_item_score_distribution.csv` | C1~C7 的 0/1/2/3 分分布。 |
| `sex_psqi_*`, `age_psqi_*` | Sex/AgeGroup 與 PSQI cutoff 的 count、percent、chi-square、t-test。 |
| `psqi_correlation.csv` | PSQI_Total、睡眠衍生欄位、C1~C7 correlation。 |
| `psqi_hierarchical_cluster.csv` | PSQI 階層式分群結果。 |
| `psqi_cluster_k_evaluation.csv` | PSQI 使用 k=2~8 的 Silhouette、Calinski-Harabasz、Davies-Bouldin、群大小比較。 |
| `psqi_cluster_profile.csv`, `psqi_raw_cluster_profile.csv` | Cluster feature profile。 |
| `psqi_named_2_cluster.csv` / `Named_2_Cluster.xlsx` | PSQI 2 群命名總表。 |
| `psqi_pca.csv`, `psqi_pca_explained_variance.csv` | PCA 座標與解釋變異。 |
| `PSQI_Analysis_Summary.xlsx` | 主要表格整合成 Excel sheets。 |
| `*.png` | result ratio、histogram、boxplot、heatmap、PCA、dendrogram、cluster ratio。 |

## Integrated 輸出格式

| 檔案 | 說明 |
|---|---|
| `integrated_scale_summary.csv` | Constitution / BDI / PSQI 的 Total、Valid、Invalid、Valid%。 |
| `integrated_common_valid_ids.csv` | 三份量表都 Valid 的共同 ID，並帶入體質 Cluster、Stage、名稱、RawCluster、dis_1~dis_4、MinDistance、NearestClusterByDistance，以及 BDI_Total、BDI_Ans、PSQI_Total、PSQI_Ans。 |
| `integrated_ids_cluster.csv` | 每個共同有效 ID 的體質 Cluster 分類，以及 dis_1~dis_4 與此分類的 Distance，距離格式為 `00.00`。 |
| `integrated_cluster_ids.csv` | 每個 Cluster 的 ID list，以及該 Cluster 內分類 Distance 的 min / mean / max，距離格式為 `00.00`。 |
| `Integrated_Statics_Summary.xlsx` | Integrated summary Excel，包含 `scale_summary`、`common_valid_ids`、`ids_cluster`、`cluster_ids` 四個 sheets。 |

## 給後續程式開發的建議

## 本版分群設定

本版依據 `RecordOriginal_3_sets.pkl` 重新比較分群品質後，採用：

```python
CON_N_CLUSTERS = 4
BDI_N_CLUSTERS = 2
PSQI_N_CLUSTERS = 2
```

原因：Constitution 的 4 群結構穩定且可解釋；BDI 與 PSQI 若使用 4 群，容易切出人數過少的小群。BDI 改為 2 群，PSQI 改為 2 群。PSQI 分群 feature 改用 `PSQI_Total + C1~C7`，避免 `ActualSleepHours`、`SleepEfficiencyPercent` 等連續欄位過度主導分群。

後續 Flask 或分群視覺化建議優先使用：

- `Data_Statics_Integrated/Constitution/constitution_hierarchical_cluster.csv`
- `Data_Statics_Integrated/Constitution/constitution_pca.csv`
- `Data_Statics_Integrated/Constitution/Named_4_Cluster.xlsx`
- `Data_Statics_Integrated/BDI/bdi_hierarchical_cluster.csv`
- `Data_Statics_Integrated/BDI/bdi_pca.csv`
- `Data_Statics_Integrated/PSQI/psqi_hierarchical_cluster.csv`
- `Data_Statics_Integrated/PSQI/psqi_pca.csv`
- `Data_Statics_Integrated/Integrated/integrated_common_valid_ids.csv`

這樣可以避免後續程式重複做分群與 PCA，也可以確保所有後續分析都使用同一份 Basic 3 sets data。
"""
    out_path = os.path.join(OUTPUT_DIR, "AnalysisStatics.md")
    label = "AnalysisStatics.md"
    if OVERWRITE_OUTPUT or not os.path.exists(out_path):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        # 同步複製一份到程式同目錄，方便閱讀與版本控管。
        with open(os.path.join(SCRIPT_DIR, "AnalysisStatics.md"), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"生成檔案({label}): {out_path}")
    else:
        print(f"已存在，略過生成({label}): {out_path}")
    


def main():
    ensure_dir(OUTPUT_DIR)
    data = load_integrated_valid_data(INPUT_PKL)
    cons_result = run_constitution(data["Constitution"])
    bdi_result = run_bdi(data["BDI"])
    psqi_result = run_psqi(data["PSQI"])
    run_integrated_outputs(cons_result, bdi_result, psqi_result)
    write_analysis_statics_md()
    print("\n完成。輸出目錄:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
