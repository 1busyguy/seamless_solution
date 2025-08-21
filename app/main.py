import streamlit as st
import os
import sys
import cv2
import numpy as np
import shutil
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
import tempfile
import subprocess
import base64

st.set_page_config(page_title="🎬 Video Stitcher", layout="wide")
st.markdown("""
    <style>
    .big-header { font-size:2rem; font-weight:700; margin-bottom:0.7em; }
    .card {background: #fafbfc; padding: 2em 1.5em; border-radius: 1em; box-shadow: 0 4px 16px 0 #0001; margin-bottom: 2em;}
    div[data-testid="column"] { gap: 0.5rem !important; }
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

# --- Small video helper (custom HTML) ---
def st_small_video(video_path, key=None):
    video_id = key or os.path.basename(video_path).replace('.', '_').replace('/', '_')
    try:
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()
        video_b64 = base64.b64encode(video_bytes).decode()
        src = f"data:video/mp4;base64,{video_b64}"
    except Exception:
        src = ""  # fallback, video will not show

    video_tag = f"""
    <video id="{video_id}" width="350" controls style="border-radius:1em; box-shadow:0 4px 16px 0 #0002;">
      <source src="{src}" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    """
    st.markdown(video_tag, unsafe_allow_html=True)

# --- Core logic from main.py ---
def extract_frame(video_path, frame_number=None, at_end=False):
    cap = cv2.VideoCapture(video_path)
    if at_end:
        frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
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
        return ssim(gray1, gray2)
    except:
        return 0

def trim_video(input_path, output_path, end_time, ffmpeg_path=None):
    ffmpeg_path = ffmpeg_path or "ffmpeg"
    cmd = [
        ffmpeg_path, '-y', '-i', input_path, '-t', str(end_time),
        '-c', 'copy', output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def get_fps_and_framecount(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frames

def find_best_match(last_frame, first_frames):
    best_match = None
    best_score = 0
    for name, frame in first_frames.items():
        score = frame_similarity(last_frame, frame)
        if score > best_score:
            best_score = score
            best_match = name
    return best_match if best_score > 0.97 else None

def find_internal_match(video_path, target_frame):
    fps, total_frames = get_fps_and_framecount(video_path)
    cap = cv2.VideoCapture(video_path)
    for i in range(total_frames - 2, -1, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        if frame_similarity(frame, target_frame) > 0.97:
            cap.release()
            return i + 1, fps
    cap.release()
    return None, fps

def stitch_ordered_from_initial(initial_path, other_paths, ffmpeg_path=None):
    pool = set(other_paths)
    sequence = []
    current = initial_path
    first_frames = {vp: extract_frame(vp, frame_number=0) for vp in pool}

    while True:
        sequence.append(current)
        last_frame = extract_frame(current, at_end=True)
        match = find_best_match(last_frame, first_frames)

        if match:
            current = match
            pool.remove(match)
            first_frames.pop(match)
        else:
            # Try to find a match within current to any first frame in pool
            found = False
            for candidate, candidate_frame in list(first_frames.items()):
                trim_point, fps = find_internal_match(current, candidate_frame)
                if trim_point:
                    duration = trim_point / fps
                    trimmed_path = current[:-4] + '_trimmed.mp4'
                    if trim_video(current, trimmed_path, duration, ffmpeg_path):
                        sequence[-1] = trimmed_path  # Replace with trimmed
                        current = candidate
                        pool.remove(candidate)
                        first_frames.pop(candidate)
                        found = True
                        break
            if not found:
                break

        if not pool:
            sequence.append(current)
            break

    return sequence

def concat_videos_ffmpeg(video_list, output_path, ffmpeg_path=None):
    ffmpeg_path = ffmpeg_path or "ffmpeg"
    list_file = 'concat_list.txt'
    with open(list_file, 'w') as f:
        for video in video_list:
            f.write(f"file '{video}'\n")

    cmd = [
        ffmpeg_path, '-y', '-f', 'concat', '-safe', '0', '-i', list_file,
        '-c', 'copy', output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(list_file)
    return result.returncode == 0

# --- Audio logic (unchanged) ---
def has_audio_stream(video_path, ffmpeg_path):
    cmd = [
        ffmpeg_path, "-i", video_path, "-hide_banner"
    ]
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    return "Audio:" in proc.stderr

def add_silent_audio(input_path, output_path, ffmpeg_path):
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

# --- Branch for Add Sound UI ---
query_params = st.query_params
addsound_path = query_params.get("addsound", None)
if addsound_path:
    addsound_path = addsound_path if isinstance(addsound_path, str) else addsound_path[0]
    st.markdown('<div class="big-header">🎵 Add Sound / AI Lip Sync</div>', unsafe_allow_html=True)
    st_small_video(addsound_path, key="addsound_video")

    st.markdown("#### 1️⃣ Add Background Music")
    music_file = st.file_uploader("Upload background music (mp3 or wav)", type=["mp3", "wav"], key="bgm")

    st.markdown("""
    <div style='
        background: #f3f4f6;
        opacity: 0.7;
        pointer-events: none;
        border-radius: 1em;
        padding: 1.5em 1em 1em 1em;
        margin-bottom: 2em;
        filter: grayscale(1);
    '>
    """, unsafe_allow_html=True)

    st.markdown("#### <span style='font-size:1.2em; font-weight:700;'>2️⃣ AI Lip Sync / Dubbing (Coming Soon)</span>", unsafe_allow_html=True)
    st.markdown("Upload audio for dubbing/lip-sync (wav, mp3)", unsafe_allow_html=True)
    st.file_uploader("Upload audio for dubbing/lip-sync (wav, mp3)", type=["wav", "mp3"], key="voice", disabled=True)
    st.checkbox("Use local Wav2Lip for lip-sync (requires wav audio)", disabled=True)
    st.text_input("HeyGen API Key (for AI video dubbing)", type="password", disabled=True)

    st.markdown("</div>", unsafe_allow_html=True)

    process_btn = st.button("Process Sound Changes")
    output_sound_path = addsound_path[:-4] + "_sound.mp4"
    output_lipsync_path = addsound_path[:-4] + "_lipsync.mp4"

    if process_btn:
        ffmpeg_path = "ffmpeg"
        # Overlay background music
        if music_file:
            music_path = os.path.join("output", "uploaded_music." + music_file.name.split(".")[-1])
            with open(music_path, "wb") as f:
                f.write(music_file.read())
            cmd = (
                f'{ffmpeg_path} -y -i "{addsound_path}" -i "{music_path}" '
                '-filter_complex "[1:a]volume=0.2[a1];[0:a][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]" '
                f'-map 0:v -map "[aout]" -shortest "{output_sound_path}"'
            )
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(output_sound_path):
                st.success("Background music added!")
                st_small_video(output_sound_path, key="sound_output_video")
                with open(output_sound_path, "rb") as file:
                    st.download_button("⬇️ Download with Background Music", file, file_name=os.path.basename(output_sound_path))
            else:
                st.error("❌ Error creating output video with sound. FFmpeg stderr:\n\n" + (result.stderr or "(no error output)"))
        if wav2lip_option and voice_file:
            wav_path = os.path.join("output", "lipsync_audio.wav")
            with open(wav_path, "wb") as f:
                f.write(voice_file.read())
            st.warning("Wav2Lip integration needed here. Insert your local pipeline code.")
        if heygen_api_key and voice_file:
            st.warning("HeyGen API integration should be done here using your API key.")

    st.markdown("---")
    if st.button("⬅️ Back to Main App"):
        st.query_params.clear()
        st.rerun()
    st.stop()

# --- Main UI ---

st.markdown('<div class="big-header">🎬 Video Stitcher</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 1️⃣ Upload Initial Video (start point)")
init_file = st.file_uploader("Upload initial video (start point)", type=["mp4", "mov", "avi"], key="init")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 2️⃣ Upload Remaining Videos (unordered)")
other_files = st.file_uploader(
    "Upload remaining videos (mp4, mov, avi). You can select multiple files.",
    type=["mp4", "mov", "avi"],
    accept_multiple_files=True,
    key="others"
)
st.markdown('</div>', unsafe_allow_html=True)

custom_audio_files = []
if init_file and other_files:
    with st.expander("🎵 Custom Audio Mix (optional, advanced)", expanded=False):
        st.markdown(
            "You can overlay up to 3 audio files (e.g. soundtrack, dialog, effects). "
            "Set the volume for each. Original video audio will be replaced by the mix."
        )
        audio1 = st.file_uploader("Soundtrack (mp3 or wav)", type=["mp3", "wav"], key="audio1")
        vol1 = st.slider("Soundtrack Volume", min_value=0, max_value=100, value=25, step=1, key="vol1")
        audio2 = st.file_uploader("Dialog (mp3 or wav)", type=["mp3", "wav"], key="audio2")
        vol2 = st.slider("Dialog Volume", min_value=0, max_value=100, value=75, step=1, key="vol2")
        audio3 = st.file_uploader("Other (mp3 or wav)", type=["mp3", "wav"], key="audio3")
        vol3 = st.slider("Other Volume", min_value=0, max_value=100, value=0, step=1, key="vol3")
        audio_mix_inputs = [
            {"file": audio1, "vol": vol1, "label": "soundtrack"},
            {"file": audio2, "vol": vol2, "label": "dialog"},
            {"file": audio3, "vol": vol3, "label": "other"},
        ]
        custom_audio_files = [item for item in audio_mix_inputs if item["file"] is not None]

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 3️⃣ Stitch Videos")
if not init_file or not other_files or len(other_files) == 0:
    st.info("Upload one initial video and at least one additional video to begin.")
else:
    if st.button("✨ Stitch Videos (Seamless + Audio)", use_container_width=True, key="stitch_btn"):
        with st.spinner("Processing your videos... this can take a minute."):
            temp_dir = os.path.join(output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            # Save initial
            init_path = os.path.join(temp_dir, init_file.name)
            with open(init_path, "wb") as f:
                f.write(init_file.read())
            # Save others
            other_paths = []
            for fobj in other_files:
                path = os.path.join(temp_dir, fobj.name)
                with open(path, "wb") as f:
                    f.write(fobj.read())
                other_paths.append(path)
            if sys.platform == "win32":
                ffmpeg_path = os.path.join(os.path.dirname(__file__), "ffmpeg", "bin", "ffmpeg.exe")
                if not os.path.isfile(ffmpeg_path):
                    ffmpeg_path = "ffmpeg"
            else:
                ffmpeg_path = "ffmpeg"
            st.info("🔎 Analyzing, matching, trimming, and ordering for seamless stitching...")

            # New logic: use main.py workflow
            stitch_list = stitch_ordered_from_initial(init_path, other_paths, ffmpeg_path=ffmpeg_path)
            st.write("🧩 Video order (auto-detected):")
            st.write([os.path.basename(p) for p in stitch_list])

            # Ensure audio on all stitched clips if needed
            files_with_audio = []
            for vpath in stitch_list:
                if not has_audio_stream(vpath, ffmpeg_path):
                    silent_path = vpath[:-4] + "_audio.mp4"
                    if add_silent_audio(vpath, silent_path, ffmpeg_path):
                        files_with_audio.append(silent_path)
                    else:
                        files_with_audio.append(vpath)
                else:
                    files_with_audio.append(vpath)

            dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_stitched = os.path.join(temp_dir, f"stitched_{dt_str}.mp4")
            final_output = os.path.join(output_dir, f"stitched_{dt_str}.mp4")
            if concat_videos_ffmpeg(files_with_audio, temp_stitched, ffmpeg_path=ffmpeg_path):
                # ---- Overlay audio AFTER main video is stitched ----
                if custom_audio_files:
                    audio_paths = []
                    mix_filters = []
                    for idx, audio in enumerate(custom_audio_files):
                        ext = os.path.splitext(audio["file"].name)[1]
                        a_path = os.path.join(temp_dir, f"audio_{audio['label']}{ext}")
                        with open(a_path, "wb") as outf:
                            outf.write(audio["file"].read())
                        mix_filters.append(f'[{idx}:a]volume={audio["vol"] / 100}[a{idx}]')
                        audio_paths.append(a_path)
                    filter_inputs_a = "".join([f'-i "{a_path}" ' for a_path in audio_paths])
                    filter_volumes = ";".join(mix_filters)
                    mix_inputs = "".join([f"[a{i}]" for i in range(len(audio_paths))])
                    mixed_audio_path = os.path.join(temp_dir, "mixed_audio.aac")
                    audio_mix_cmd = (
                        f'{ffmpeg_path} {filter_inputs_a}-filter_complex "{filter_volumes};{mix_inputs}amix=inputs={len(audio_paths)}:duration=longest[mixed]" '
                        f'-map "[mixed]" -ac 2 -c:a aac -y "{mixed_audio_path}"'
                    )
                    subprocess.run(audio_mix_cmd, shell=True, capture_output=True, text=True)
                    replace_audio_cmd = (
                        f'"{ffmpeg_path}" -y -i "{temp_stitched}" -i "{mixed_audio_path}" '
                        f'-c:v copy -map 0:v:0 -map 1:a:0 -shortest "{final_output}"'
                    )
                    replace_result = subprocess.run(replace_audio_cmd, shell=True, capture_output=True, text=True)
                    if replace_result.returncode == 0 and os.path.isfile(final_output):
                        st.success("✅ Done! Download your stitched video below:")
                        col1, col2, col3 = st.columns([2, 3, 2])
                        with col2:
                            st.video(final_output, format="video/mp4", start_time=0)
                            with open(final_output, "rb") as file:
                                st.download_button("⬇️ Download Video", file, file_name=os.path.basename(final_output))
                        if st.button("🎵 Add Sound", key=f"add_sound_{os.path.basename(final_output)}"):
                            st.query_params["addsound"] = final_output
                            st.rerun()
                        st.write(f"Or find it in the **output** folder.")
                    else:
                        error_msg = replace_result.stderr or "No FFmpeg stderr captured."
                        st.error(f"⚠️ FFmpeg audio replace error:\n\n```\n{error_msg}\n```")
                else:
                    shutil.copy(temp_stitched, final_output)
                    st.success("✅ Done! Download your stitched video below:")
                    col1, col2, col3 = st.columns([2, 3, 2])
                    with col2:
                        st.video(final_output, format="video/mp4", start_time=0)
                        with open(final_output, "rb") as file:
                            st.download_button("⬇️ Download Video", file, file_name=os.path.basename(final_output))
                    if st.button("🎵 Add Sound", key=f"add_sound_{os.path.basename(final_output)}"):
                        st.query_params["addsound"] = final_output
                        st.rerun()
                    st.write(f"Or find it in the **output** folder.")
            else:
                st.error("⚠️ FFmpeg error while stitching videos.")
            shutil.rmtree(temp_dir, ignore_errors=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Output History Section ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 4️⃣ Output History")
videos = [f for f in os.listdir(output_dir) if f.lower().endswith(".mp4")]
if videos:
    videos = sorted(videos, reverse=True)[:8]
    for i in range(0, len(videos), 4):
        cols = st.columns(4)
        for j, idx in enumerate(range(i, min(i+4, len(videos)))):
            vid = videos[idx]
            vpath = os.path.join(output_dir, vid)
            with cols[j]:
                st.video(vpath, format="video/mp4", start_time=0)
                with open(vpath, "rb") as file:
                    st.download_button("⬇️ Download again", file, file_name=vid)
                if st.button("🎵 Add Sound", key=f"add_sound_hist_{vid}"):
                    st.query_params["addsound"] = vpath
                    st.rerun()
else:
    st.write("No output videos yet. Your stitched results will appear here!")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.write("All stitched videos are saved in the **output** folder next to this app.")
