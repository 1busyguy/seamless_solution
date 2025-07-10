import streamlit as st
import os
import sys
import cv2
import numpy as np
import shutil
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from streamlit_sortables import sort_items
import tempfile

st.set_page_config(page_title="🎬 Video Stitcher", layout="wide")
st.markdown("""
    <style>
    .big-header { font-size:2rem; font-weight:700; margin-bottom:0.7em; }
    .card {background: #fafbfc; padding: 2em 1.5em; border-radius: 1em; box-shadow: 0 4px 16px 0 #0001; margin-bottom: 2em;}
    div[data-testid="column"] { gap: 0.5rem !important; }
    /* Center content, subtle box */
    .block-container {
        max-width: 1200px;
        margin: 2.5em auto 1.5em auto;
        background: #fff;
        border-radius: 1.5em;
        box-shadow: 0 8px 40px 0 #0001;
        padding: 2.5em 2em;
    }
    </style>
""", unsafe_allow_html=True)

output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)

# --- Helper functions ---
def extract_frame_img(video_bytes):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    ret, frame = cap.read()
    cap.release()
    os.unlink(tmp_path)
    if ret and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)
        # No .thumbnail()! Keep full size for clarity, shrink only in st.image
        return pil_image
    return None

def extract_frame(video_path, frame_number=None, at_end=False):
    cap = cv2.VideoCapture(video_path)
    if at_end:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = total - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def frame_similarity(frame1, frame2):
    try:
        frame1 = cv2.resize(frame1, (320, 240))
        frame2 = cv2.resize(frame2, (320, 240))
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        sim = ssim(gray1, gray2)
        return sim
    except Exception:
        return 0

def trim_video(input_path, output_path, end_time, ffmpeg_path):
    cmd = [
        ffmpeg_path, "-y", "-i", input_path, "-t", str(end_time),
        "-c", "copy", output_path
    ]
    result = os.system(" ".join(cmd))
    return os.path.exists(output_path)

def get_fps_and_framecount(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frames

def has_audio_stream(video_path, ffmpeg_path):
    cmd = [
        ffmpeg_path, "-i", video_path, "-hide_banner"
    ]
    import subprocess
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    return "Audio:" in proc.stderr

def add_silent_audio(input_path, output_path, ffmpeg_path):
    import subprocess
    # Get duration of input video
    probe_cmd = [
        ffmpeg_path, "-i", input_path, "-hide_banner"
    ]
    proc = subprocess.run(probe_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    import re
    match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", proc.stderr)
    if match:
        h, m, s = match.groups()
        duration = float(h) * 3600 + float(m) * 60 + float(s)
        if duration < 0.1:
            duration = 0.2  # for very short videos
    else:
        duration = 1.0  # fallback
    cmd = [
        ffmpeg_path, "-y", "-i", input_path,
        "-f", "lavfi", "-t", f"{duration:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-shortest", "-c:v", "copy", "-c:a", "aac", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.exists(output_path)

# --- UI Layout ---
st.markdown('<div class="big-header">🎬 Video Stitcher</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 1️⃣ Upload Video Files")
uploaded_files = st.file_uploader(
    "Choose videos (mp4, mov, avi, mkv). You can upload more than one.",
    type=['mp4', 'mov', 'avi', 'mkv'],
    accept_multiple_files=True,
    key="uploader"
)
st.markdown('</div>', unsafe_allow_html=True)

# --- Thumbnails + drag and drop reordering ---
if uploaded_files:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 2️⃣ Preview & Arrange Your Videos")
    st.write("🖼️ **Preview thumbnails below:**")
    thumb_filenames = []

    # Display 4 per row using st.columns
    for i in range(0, len(uploaded_files), 4):
        cols = st.columns(4)
        for j, idx in enumerate(range(i, min(i+4, len(uploaded_files)))):
            file = uploaded_files[idx]
            thumb = extract_frame_img(file.getbuffer())
            label = f"{idx+1}. {file.name}"
            if thumb:
                cols[j].image(thumb, caption=label, width=150)
            thumb_filenames.append(file.name)

    st.write("**Drag and drop to arrange order below:**")
    new_order = sort_items(thumb_filenames, direction="horizontal")
    ordered_files = [next(f for f in uploaded_files if f.name == fname) for fname in new_order]
    st.markdown('</div>', unsafe_allow_html=True)
else:
    ordered_files = []

# --- Stitch Button ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 3️⃣ Stitch Videos")
if not uploaded_files:
    st.info("Upload at least two videos to begin.")
elif len(ordered_files) < 2:
    st.info("Upload at least two videos to enable stitching.")
else:
    if st.button("✨ Stitch Videos (Seamless + Audio)", use_container_width=True, key="stitch_btn"):
        with st.spinner("Processing your videos... this can take a minute."):
            temp_dir = os.path.join(output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_files = []
            # Save uploads in new order
            for i, file in enumerate(ordered_files):
                ext = os.path.splitext(file.name)[1]
                temp_path = os.path.join(temp_dir, f"vid{i:03d}{ext}")
                with open(temp_path, "wb") as out:
                    out.write(file.read())
                temp_files.append(temp_path)
            # Use portable ffmpeg in bin/ if available
            if sys.platform == "win32":
                ffmpeg_path = os.path.join(os.path.dirname(__file__), "ffmpeg", "bin", "ffmpeg.exe")
                if not os.path.isfile(ffmpeg_path):
                    ffmpeg_path = "ffmpeg"
            else:
                ffmpeg_path = "ffmpeg"
            st.info("🔎 Analyzing, trimming, and ensuring audio for seamless stitching...")
            processed_files = []
            progress = st.progress(0)
            for idx in range(len(temp_files)-1):
                v1, v2 = temp_files[idx], temp_files[idx+1]
                last_frame_v1 = extract_frame(v1, at_end=True)
                first_frame_v2 = extract_frame(v2, frame_number=0)
                sim = frame_similarity(last_frame_v1, first_frame_v2)
                v1_to_use = v1
                # If not similar, scan back for a match and trim
                if sim <= 0.97:
                    fps, frames = get_fps_and_framecount(v1)
                    trim_idx = frames
                    found = False
                    cap = cv2.VideoCapture(v1)
                    for i in range(frames-2, -1, -1):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                        ret, frame = cap.read()
                        if not ret: continue
                        if frame_similarity(frame, first_frame_v2) > 0.97:
                            trim_idx = i + 1
                            found = True
                            break
                    cap.release()
                    if found and trim_idx < frames:
                        end_time = trim_idx / fps
                        trimmed_path = v1[:-4] + "_trimmed.mp4"
                        ok = trim_video(v1, trimmed_path, end_time, ffmpeg_path)
                        if ok:
                            v1_to_use = trimmed_path
                # After possible trimming, ensure audio
                if not has_audio_stream(v1_to_use, ffmpeg_path):
                    silent_v1 = v1_to_use[:-4] + "_audio.mp4"
                    if add_silent_audio(v1_to_use, silent_v1, ffmpeg_path):
                        v1_to_use = silent_v1
                processed_files.append(v1_to_use)
                progress.progress(int(100 * (idx + 1) / (len(temp_files))))
            # Last file: always patch if needed
            last_file = temp_files[-1]
            last_to_use = last_file
            if not has_audio_stream(last_file, ffmpeg_path):
                silent_last = last_file[:-4] + "_audio.mp4"
                if add_silent_audio(last_file, silent_last, ffmpeg_path):
                    last_to_use = silent_last
            processed_files.append(last_to_use)
            st.info("🔗 Stitching videos together (almost done)...")
            dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_video = os.path.join(output_dir, f"stitched_{dt_str}.mp4")
            # -- CONCAT WITH AUDIO --
            filter_file_list = ""
            for f in processed_files:
                filter_file_list += f'-i "{f}" '
            filter_inputs = "".join([f"[{i}:v:0][{i}:a:0]" for i in range(len(processed_files))])
            filter_complex = f'{filter_inputs}concat=n={len(processed_files)}:v=1:a=1[outv][outa]'
            command = (
                f'"{ffmpeg_path}" {filter_file_list}-filter_complex "{filter_complex}" '
                f'-map "[outv]" -map "[outa]" -y "{output_video}"'
            )
            try:
                import subprocess
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                if result.returncode == 0 and os.path.isfile(output_video):
                    st.success("✅ Done! Download your stitched video below:")
                    col1, col2, col3 = st.columns([2, 3, 2])  # Center and shrink video
                    with col2:
                        st.video(output_video, format="video/mp4", start_time=0)
                        with open(output_video, "rb") as file:
                            st.download_button("⬇️ Download Video", file, file_name=f"stitched_{dt_str}.mp4")
                    st.write(f"Or find it in the **output** folder.")
                else:
                    st.error(f"⚠️ FFmpeg error:\n\n{result.stderr}")
            except Exception as e:
                st.error(f"Error stitching videos: {e}")
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Output History Section ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 4️⃣ Output History")
videos = [f for f in os.listdir(output_dir) if f.lower().endswith(".mp4")]
if videos:
    videos = sorted(videos, reverse=True)[:8]  # Show up to 8 recent videos (adjust as needed)
    for i in range(0, len(videos), 4):
        cols = st.columns(4)
        for j, idx in enumerate(range(i, min(i+4, len(videos)))):
            vid = videos[idx]
            vpath = os.path.join(output_dir, vid)
            with cols[j]:
                st.video(vpath, format="video/mp4", start_time=0)
                with open(vpath, "rb") as file:
                    st.download_button("⬇️ Download again", file, file_name=vid)
else:
    st.write("No output videos yet. Your stitched results will appear here!")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.write("All stitched videos are saved in the **output** folder next to this app.")
