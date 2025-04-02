# predict_and_record.py

import os
import re
import requests
import pandas as pd
import cv2
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from time import sleep
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# === 資料夾與檔案路徑設定 ===
BASE_FOLDER = r"C:\\Users\\王詩偉\\Desktop\\專題"
ORIGINAL_FILES_FOLDER = os.path.join(BASE_FOLDER, "下載原始檔案")
PROCESSED_FOLDER = os.path.join(BASE_FOLDER, "整理後檔案")
MERGED_FOLDER = os.path.join(BASE_FOLDER, "合併的")
MAPPING_FILE_PATH = os.path.join(BASE_FOLDER, "Gantry_Road_Mapping.xlsx")
CCTV_MAPPING_FILE = os.path.join(BASE_FOLDER, "Gantry_Road_Mapping_Pro.xlsx")
TRAINING_DATA_PATH = r"C:\\Users\\王詩偉\\Downloads\\Recalculated_Median_Speed__Grouped_by_5_(1).csv"

os.makedirs(ORIGINAL_FILES_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(MERGED_FOLDER, exist_ok=True)

# === 定義函數區 ===
def same_direction(road_id1, road_id2):
    pattern = r"([NS])$"
    match1 = re.search(pattern, road_id1)
    match2 = re.search(pattern, road_id2)
    return match1 and match2 and match1.group() == match2.group()

def record_cctv_video(cctv_url, save_path, duration=30):
    cap = cv2.VideoCapture(cctv_url)
    if not cap.isOpened():
        print(f"❌ 無法開啟影片來源：{cctv_url}")
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    width = int(cap.get(3))
    height = int(cap.get(4))
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    start_time = datetime.now()
    while (datetime.now() - start_time).seconds < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()
    print(f"✅ 已錄製：{save_path}")

def record_high_risk_cctv_from_result(high_risk_df, model_name, cctv_mapping_df):
    gantry_to_cctv = dict(zip(cctv_mapping_df["Gantry ID"], cctv_mapping_df["VideoStreamURL"]))
    for _, row in high_risk_df.iterrows():
        for col in ['RoadID_1', 'RoadID_2']:
            gantry = row[col]
            cctv_url = gantry_to_cctv.get(gantry)
            if cctv_url:
                filename = f"{model_name}_{gantry}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                save_path = os.path.join(BASE_FOLDER, "CCTV_Recordings", filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                record_cctv_video(cctv_url, save_path)
            else:
                print(f"[警告] 找不到 {gantry} 對應的 CCTV URL")

def generate_base_url(base_time):
    current_date = base_time.strftime("%Y%m%d")
    current_hour = base_time.strftime("%H")
    return f"https://tisvcloud.freeway.gov.tw/history/TDCS/M05A/{current_date}/{current_hour}/"

def get_latest_csv(base_url):
    response = requests.get(base_url)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    csv_files = [a['href'] for a in soup.find_all('a') if a['href'].endswith('.csv')]
    if not csv_files:
        return None
    def extract_timestamp(filename):
        try:
            parts = filename.replace('.csv', '').split('_')
            return datetime.strptime(parts[2] + parts[3], "%Y%m%d%H%M%S")
        except Exception:
            return datetime.min
    latest_file = max(csv_files, key=extract_timestamp)
    return urljoin(base_url, latest_file)

def download_file(file_url, save_folder):
    filename = os.path.join(save_folder, file_url.split("/")[-1])
    if os.path.exists(filename):
        print(f"📂 檔案已存在，跳過下載：{filename}")
        return filename
    response = requests.get(file_url)
    response.raise_for_status()
    with open(filename, 'wb') as file:
        file.write(response.content)
    print(f"✅ 成功下載：{filename}")
    return filename

def process_file(file_path):
    try:
        data = pd.read_csv(file_path, header=None)
        data.columns = ["Time", "RoadID_1", "RoadID_2", "VehicleType", "Speed", "Flow"]
        pivoted = data.pivot_table(index=["RoadID_1", "RoadID_2"], columns="VehicleType",
                                   values=["Speed", "Flow"], aggfunc="first", fill_value=0)
        pivoted.columns = ['_'.join(map(str, col)) for col in pivoted.columns]
        pivoted.reset_index(inplace=True)
        final_order = ["RoadID_1", "RoadID_2"] + \
                      [f"Flow_{t}" for t in [31, 32, 41, 42, 5]] + \
                      [f"Speed_{t}" for t in [31, 32, 41, 42, 5]]
        final = pivoted.reindex(columns=final_order, fill_value=0)
        output_file = os.path.join(PROCESSED_FOLDER, f"Processed_{os.path.basename(file_path)}.xlsx")
        final.to_excel(output_file, index=False)
        print(f"✅ 整理完成：{output_file}")
        return output_file
    except Exception as e:
        print(f"❌ 整理失敗：{e}")

def merge_latest_files():
    files = sorted([
        os.path.join(PROCESSED_FOLDER, f)
        for f in os.listdir(PROCESSED_FOLDER)
        if f.endswith(".xlsx") and not f.startswith("~$")
    ], key=os.path.getmtime, reverse=True)
    if len(files) < 5:
        print("⚠️ 整理後檔案不足 5 筆，略過預測")
        return None
    latest = files[:5]
    dfs = [pd.read_excel(f) for f in latest]
    merged = pd.concat(dfs)
    merged = merged.groupby(["RoadID_1", "RoadID_2"], as_index=False).agg({
        "Flow_31": "sum", "Flow_32": "sum", "Flow_41": "sum", "Flow_42": "sum", "Flow_5": "sum",
        "Speed_31": "median", "Speed_32": "median", "Speed_41": "median", "Speed_42": "median", "Speed_5": "median"
    })
    output_file = os.path.join(MERGED_FOLDER, "merged_latest_5.xlsx")
    merged.to_excel(output_file, index=False)
    print(f"✅ 合併完成：{output_file}")
    return output_file

def train_and_predict():
    downloaded_file = None
    for hour_back in range(3):
        check_time = datetime.now() - pd.Timedelta(hours=hour_back)
        base_url = generate_base_url(check_time)
        latest_csv_url = get_latest_csv(base_url)
        if latest_csv_url:
            downloaded_file = download_file(latest_csv_url, ORIGINAL_FILES_FOLDER)
            process_file(downloaded_file)
            break
        else:
            print(f"⚠️ 無法取得 {check_time.strftime('%Y-%m-%d %H')} 時段資料，回溯中...")
    if not downloaded_file:
        print("❌ 回溯三次皆未取得資料，跳過本輪預測。")
        return

    merged_file_path = merge_latest_files()
    if not merged_file_path:
        return

    df = pd.read_csv(TRAINING_DATA_PATH)
    X = df.drop(columns=['Y'])
    y = df['Y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }

    df_cctv_mapping = pd.read_excel(CCTV_MAPPING_FILE)
    df_gantry_mapping = pd.read_excel(MAPPING_FILE_PATH)
    gantry_dict = dict(zip(df_gantry_mapping["Gantry ID"], df_gantry_mapping["Road Segment"]))

    merged_df = pd.read_excel(merged_file_path).fillna(0)
    feature_cols = [col for col in merged_df.columns if col not in ["RoadID_1", "RoadID_2"]]

    for name, model in models.items():
        model.fit(X_train_resampled, y_train_resampled)
        risk_scores = model.predict_proba(merged_df[feature_cols])[:, 1]
        merged_df['Risk_Score'] = risk_scores
        high_risk = merged_df[merged_df['Risk_Score'] > 0.8][['RoadID_1', 'RoadID_2', 'Risk_Score']]
        high_risk["RoadID_1_Segment"] = high_risk["RoadID_1"].map(gantry_dict)
        high_risk["RoadID_2_Segment"] = high_risk["RoadID_2"].map(gantry_dict)
        print(f"\n=== 高風險門架 (模型: {name}) ===")
        print(high_risk.head(10))
        record_high_risk_cctv_from_result(high_risk, name, df_cctv_mapping)

# === 程式主入口點 ===
if __name__ == "__main__":
    train_and_predict()
