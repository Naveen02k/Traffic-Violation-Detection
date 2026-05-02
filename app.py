"""
Final Streamlit UI - Traffic Violation Detection System
=======================================================

Features:
- Upload traffic video
- Select traffic direction
- ROI-based automatic signal color detection
- Processed annotated video
- Violation images
- Evidence clips
- CSV report
- Dashboard analytics with graphs

Run:
    streamlit run app.py
"""

import sys
import shutil
import subprocess
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Traffic Violation Detection System",
    page_icon="🚦",
    layout="wide",
)


PROJECT_DIR = Path(__file__).parent
UPLOAD_DIR = PROJECT_DIR / "uploaded_videos"
OUTPUT_DIR = PROJECT_DIR / "output_ui"
MAIN_SCRIPT = PROJECT_DIR / "main.py"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


st.markdown(
    """
    <style>
        .main-title {
            font-size: 44px;
            font-weight: 800;
            margin-bottom: 0px;
        }
        .subtitle {
            font-size: 17px;
            color: #9ca3af;
            margin-bottom: 25px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_output():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)


def save_uploaded_video(uploaded_file):
    video_path = UPLOAD_DIR / uploaded_file.name
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    return video_path


def get_first_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def run_detection(video_path, confidence, allowed_direction, signal_mode, signal_roi, yellow_is_stop):
    clean_output()

    command = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--source",
        str(video_path),
        "--output",
        str(OUTPUT_DIR),
        "--conf",
        str(confidence),
        "--allowed-direction",
        str(allowed_direction),
        "--no-display",
    ]

    if signal_mode == "roi" and signal_roi:
        command.extend(["--signal-mode", "roi"])
        command.extend(["--signal-roi", signal_roi])
        if yellow_is_stop:
            command.append("--yellow-is-stop")

    return subprocess.run(command, capture_output=True, text=True)


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def make_browser_playable(video_path: Path):
    if not video_path.exists():
        return video_path

    if not ffmpeg_available():
        return video_path

    converted_dir = OUTPUT_DIR / "browser_videos"
    converted_dir.mkdir(exist_ok=True)

    out_path = converted_dir / video_path.name

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-acodec",
        "aac",
        str(out_path),
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    return video_path


def unique_paths(paths):
    seen = set()
    result = []
    for path in paths:
        try:
            key = str(path.resolve()).lower()
        except Exception:
            key = str(path).lower()
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def read_report():
    csv_path = OUTPUT_DIR / "violations.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path, encoding="utf-8")
    except Exception:
        return pd.DataFrame()


def show_summary_cards():
    df = read_report()

    total = 0
    signal = 0
    wrong = 0
    helmet = 0
    avg_risk = 0

    if not df.empty and "violation_type" in df.columns:
        col = df["violation_type"].astype(str).str.lower()
        total = len(df)
        signal = len(df[col.str.contains("signal", na=False)])
        wrong = len(df[col.str.contains("wrong", na=False)])
        helmet = len(df[col.str.contains("helmet", na=False)])

        if "risk_score" in df.columns:
            avg_risk = round(df["risk_score"].fillna(0).mean(), 2)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Violations", total)
    c2.metric("Signal Jump", signal)
    c3.metric("Wrong Way", wrong)
    c4.metric("Helmet Violation", helmet)
    c5.metric("Avg Risk Score", avg_risk)


def show_dashboard():
    st.subheader("📊 Dashboard Analytics")

    df = read_report()

    if df.empty:
        st.info("No dashboard data available. Run detection first.")
        return

    if "violation_type" not in df.columns:
        st.warning("CSV does not contain violation_type column.")
        return

    df["violation_type"] = df["violation_type"].astype(str)

    left, right = st.columns(2)

    with left:
        st.markdown("### Violation Count by Type")
        count_df = df["violation_type"].value_counts().reset_index()
        count_df.columns = ["Violation Type", "Count"]
        st.bar_chart(count_df, x="Violation Type", y="Count")

    with right:
        st.markdown("### Risk Score by Vehicle")
        if "vehicle_id" in df.columns and "risk_score" in df.columns:
            risk_df = (
                df.groupby("vehicle_id")["risk_score"]
                .max()
                .reset_index()
                .sort_values("risk_score", ascending=False)
            )
            risk_df["vehicle_id"] = risk_df["vehicle_id"].astype(str)
            st.bar_chart(risk_df, x="vehicle_id", y="risk_score")
        else:
            st.info("Risk score data not available.")

    st.divider()

    left2, right2 = st.columns(2)

    with left2:
        st.markdown("### Violations Over Frames")
        if "frame" in df.columns:
            frame_df = df[["frame"]].copy()
            frame_df["count"] = 1
            frame_df = frame_df.sort_values("frame")
            st.line_chart(frame_df, x="frame", y="count")
        else:
            st.info("Frame data not available.")

    with right2:
        st.markdown("### Vehicle Type Distribution")
        if "vehicle_type" in df.columns:
            vehicle_df = df["vehicle_type"].value_counts().reset_index()
            vehicle_df.columns = ["Vehicle Type", "Count"]
            st.bar_chart(vehicle_df, x="Vehicle Type", y="Count")
        else:
            st.info("Vehicle type data not available.")

    st.divider()
    st.markdown("### High Risk Vehicles")
    if "vehicle_id" in df.columns and "risk_score" in df.columns:
        high_risk = (
            df.groupby(["vehicle_id", "vehicle_type"], dropna=False)["risk_score"]
            .max()
            .reset_index()
            .sort_values("risk_score", ascending=False)
        )
        st.dataframe(high_risk, use_container_width=True)
    else:
        st.info("High risk vehicle table not available.")


def show_violation_images():
    possible_folders = {
        "Signal Jump": ["signal_jump", "SIGNAL_JUMP"],
        "Wrong Way": ["wrong_way", "WRONG_WAY"],
        "Helmet Violation": ["helmet_violation", "HELMET_VIOLATION"],
    }

    for title, folder_names in possible_folders.items():
        st.subheader(title)
        images = []

        for folder_name in folder_names:
            folder = OUTPUT_DIR / folder_name
            if folder.exists():
                images.extend(list(folder.glob("*.jpg")))
                images.extend(list(folder.glob("*.jpeg")))
                images.extend(list(folder.glob("*.png")))

        images = unique_paths(images)

        if not images:
            st.write("No images found.")
            continue

        cols = st.columns(3)
        for idx, img_path in enumerate(images):
            with cols[idx % 3]:
                st.image(str(img_path), caption=img_path.name, use_container_width=True)


def show_evidence_clips():
    st.subheader("Evidence Clips")

    folders = [OUTPUT_DIR / "evidence_clips", OUTPUT_DIR / "EVIDENCE_CLIPS"]
    clips = []

    for folder in folders:
        if folder.exists():
            clips.extend(list(folder.glob("*.mp4")))

    clips = unique_paths(clips)

    if not clips:
        st.write("No evidence clips found.")
        return

    if not ffmpeg_available():
        st.warning("FFmpeg is not available. Clips are saved, but browser playback may fail.")

    for clip in clips:
        st.write(f"**{clip.name}**")
        playable_clip = make_browser_playable(clip)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.video(str(playable_clip))


def show_csv_report():
    st.subheader("CSV Violation Report")
    df = read_report()

    if df.empty:
        st.write("No CSV data found yet.")
        return

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV Report",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="violations_report.csv",
        mime="text/csv",
    )


st.markdown('<div class="main-title">🚦 Traffic Violation Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload video, run detection, view violations, evidence clips, CSV report, and dashboard analytics.</div>',
    unsafe_allow_html=True,
)

if not MAIN_SCRIPT.exists():
    st.error("main.py not found. Keep app.py and main.py in the same project folder.")
    st.stop()


with st.sidebar:
    st.header("Upload & Settings")

    uploaded_video = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov", "mkv"])

    confidence = st.slider("YOLO Confidence", min_value=0.10, max_value=0.90, value=0.45, step=0.05)

    allowed_direction = st.selectbox(
        "Allowed Traffic Direction",
        options=[0, 90, 180, -90],
        format_func=lambda x: {
            0: "Right →",
            90: "Down ↓",
            180: "Left ←",
            -90: "Up ↑",
        }[x],
    )

    st.divider()
    st.subheader("Traffic Signal Detection")

    use_signal_roi = st.checkbox("Auto detect signal from ROI", value=True)
    yellow_is_stop = st.checkbox("Treat yellow as stop", value=True)

    run_button = st.button("Run Detection", type="primary")


if uploaded_video is None:
    st.info("Upload a traffic video from the sidebar.")
    st.stop()


video_path = save_uploaded_video(uploaded_video)

signal_mode = "cycle"
signal_roi = ""

first_frame = get_first_frame(video_path)

if use_signal_roi and first_frame is not None:
    h, w = first_frame.shape[:2]

    st.sidebar.caption("Adjust ROI around the traffic signal in the first frame.")

    default_x1 = int(w * 0.82)
    default_y1 = int(h * 0.08)
    default_x2 = int(w * 0.98)
    default_y2 = int(h * 0.40)

    roi_x1 = st.sidebar.slider("Signal ROI x1", 0, w - 1, min(default_x1, w - 1))
    roi_y1 = st.sidebar.slider("Signal ROI y1", 0, h - 1, min(default_y1, h - 1))
    roi_x2 = st.sidebar.slider("Signal ROI x2", 1, w, min(default_x2, w))
    roi_y2 = st.sidebar.slider("Signal ROI y2", 1, h, min(default_y2, h))

    if roi_x2 > roi_x1 and roi_y2 > roi_y1:
        signal_mode = "roi"
        signal_roi = f"{roi_x1},{roi_y1},{roi_x2},{roi_y2}"


st.subheader("Uploaded Video")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.video(str(video_path))


if signal_mode == "roi" and first_frame is not None:
    preview = first_frame.copy()
    x1, y1, x2, y2 = [int(v) for v in signal_roi.split(",")]

    cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 255, 0), 3)
    cv2.putText(preview, "SIGNAL ROI", (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

    st.subheader("Signal ROI Preview")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(preview_rgb, use_container_width=True)


st.divider()


if run_button:
    with st.spinner("Processing video... please wait"):
        result = run_detection(
            video_path=video_path,
            confidence=confidence,
            allowed_direction=allowed_direction,
            signal_mode=signal_mode,
            signal_roi=signal_roi,
            yellow_is_stop=yellow_is_stop,
        )

    if result.returncode != 0:
        st.error("Detection failed.")
        st.code(result.stderr)
        st.stop()

    st.success("Detection completed successfully.")

    with st.expander("Show Processing Logs"):
        st.code(result.stdout + "\n" + result.stderr)


show_summary_cards()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Processed Video", "Dashboard", "Violation Images", "Evidence Clips", "CSV Report"]
)

with tab1:
    st.subheader("Processed Video")

    possible_videos = unique_paths([OUTPUT_DIR / "output_annotated.mp4", OUTPUT_DIR / "output_video.mp4"])
    found = False

    for video in possible_videos:
        if video.exists():
            playable_video = make_browser_playable(video)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.video(str(playable_video))
            found = True
            break

    if not found:
        st.info("Processed video not found. Run detection first.")

with tab2:
    show_dashboard()

with tab3:
    show_violation_images()

with tab4:
    show_evidence_clips()

with tab5:
    show_csv_report()
