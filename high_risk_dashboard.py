import streamlit as st
import pandas as pd
import os
import glob
import subprocess

# === 資料夾與檔案設定 ===
BASE_FOLDER = os.getcwd()
VIDEO_FOLDER = os.path.join(BASE_FOLDER, "CCTV_Recordings")
SNAPSHOT_FOLDER = os.path.join(BASE_FOLDER, "YOLO_Snapshots")
FFMPEG_EXE = os.path.join(BASE_FOLDER, "ffmpeg.exe")
FILE_PATTERN = "High_Risk_Gantry_With_Segments_*.xlsx"

# === 自動轉檔函數 ===
def convert_video_to_h264(input_path):
    output_path = input_path.replace(".mp4", "_converted.mp4")
    if os.path.exists(output_path):
        print(f"✅ 已轉檔過：{output_path}")
        return output_path
    try:
        command = [
            FFMPEG_EXE,
            "-y",
            "-i", input_path,
            "-vcodec", "libx264",
            "-acodec", "aac",
            output_path
        ]
        subprocess.run(command, check=True)
        print(f"✅ 成功轉檔為 H.264：{output_path}")
        return output_path
    except Exception as e:
        print(f"❌ 轉檔失敗：{e}")
        return input_path

# === 找最新模型預測檔案 ===
def get_latest_model_file(model_name):
    pattern = os.path.join(BASE_FOLDER, f"High_Risk_Gantry_With_Segments_{model_name}.xlsx")
    return pattern if os.path.exists(pattern) else None

# === 載入資料 ===
def load_latest_data(model_name):
    path = get_latest_model_file(model_name)
    if not path:
        return None, None
    df = pd.read_excel(path)
    return df, model_name

# === 找影片檔（優先轉檔後的） ===
def find_video(gantry_id, model_name):
    gantry_id = gantry_id.strip()
    converted_pattern = os.path.join(VIDEO_FOLDER, f"{model_name}_{gantry_id}_*_converted.mp4")
    raw_pattern = os.path.join(VIDEO_FOLDER, f"{model_name}_{gantry_id}_*.mp4")
    converted_files = glob.glob(converted_pattern)
    if converted_files:
        print("[DEBUG] 找到轉檔影片：", converted_files[0])
        return converted_files[0]
    raw_files = [f for f in glob.glob(raw_pattern) if "_converted" not in f]
    if raw_files:
        print("[DEBUG] 找到原始影片並嘗試轉檔：", raw_files[0])
        return convert_video_to_h264(raw_files[0])
    print("[DEBUG] 找不到任何影片")
    return None

# === 找事故截圖 ===
def find_snapshot(gantry_id, model_name):
    gantry_id = gantry_id.strip()
    pattern = os.path.join(SNAPSHOT_FOLDER, f"{model_name}_{gantry_id}_*.jpg")
    files = glob.glob(pattern)
    return files[0] if files else None

# === Streamlit 主畫面 ===
st.set_page_config(page_title="高風險門架 Dashboard", layout="wide")
st.title("🚨 高風險門架 Dashboard")

model_options = ["LogisticRegression", "Random Forest", "XGBoost"]
selected_model = st.selectbox("選擇要顯示的模型結果：", model_options)

df, model_name = load_latest_data(selected_model)

if df is None:
    st.error(f"⚠️ 找不到模型 {selected_model} 的預測結果，請確認是否已執行預測。")
else:
    st.subheader(f"模型來源：{model_name}")
    df = df[df['Risk_Score'] >= 0.8].copy()
    df['Segment'] = df['RoadID_1_Segment'] + " → " + df['RoadID_2_Segment']
    df['Risk_Score_Label'] = (df['Risk_Score'] * 100).round(1).astype(str) + "%"

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("📋 高風險路段清單")
        selected_index = st.radio(
            "請選擇一個路段：",
            options=df.index,
            format_func=lambda i: f"{df.loc[i, 'Segment']} ({df.loc[i, 'Risk_Score_Label']})"
        )

    with right_col:
        row = df.loc[selected_index]
        st.subheader(f"📍 路段：{row['Segment']}")
        st.markdown(f"**風險分數**：{row['Risk_Score_Label']}")

        st.markdown("---")
        st.markdown("#### 🎥 事故影片預覽")
        video_path = find_video(row['RoadID_1'], model_name)
        st.write("🔍 偵測影片路徑：", video_path)

        if video_path and os.path.exists(video_path):
            st.video(video_path)
        else:
            st.warning("⚠️ 找不到影片，改播放測試影片")
            st.video("https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4")

        st.markdown("---")
        st.markdown("#### 🖼️ 事故擷取畫面")
        snapshot_path = find_snapshot(row['RoadID_1'], model_name)
        if snapshot_path:
            st.image(snapshot_path, caption="偵測到的事故畫面")
        else:
            st.info("⚠️ 尚無事故擷圖可顯示")

        st.markdown("---")
        st.markdown("#### 🎞️ 可用影片清單")
        available_videos = glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4"))
        for vid in available_videos:
            st.text(os.path.basename(vid)) 






 
