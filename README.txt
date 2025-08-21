VIDEO STITCHER

Quick Start

1. Unzip everything into a folder (or clone from GitHub).
2. Make sure you have Python 3.11.x installed.
   Get Python: https://www.python.org/downloads/release/python-3110/
3. Double-click "run.bat" (Windows). The app will set up a virtual environment and open in your browser.

How to Use

1. **Upload Videos:**
   - Upload your initial (starting) video using the first uploader.
   - Upload all other unordered videos using the second uploader. The app will figure out the correct order for stitching automatically.

2. **Stitch Videos:**
   - Click "Stitch Videos" and wait while the app analyzes, trims, and stitches your files in the best order.
   - Download your stitched video or find it in the `app/output` folder.

3. **Output History:**
   - All previously created stitched videos appear in the "Output History" section.
   - You can play or download any past video at any time.

4. **Add Sound:**
   - After stitching, click the "Add Sound" button (shown under the latest video and in output history) to open the sound page.
   - On the Add Sound page, you can overlay background music to your video.
   - The AI Lip Sync / Dubbing section is grayed out and not yet available.

Troubleshooting

- If videos do not play, make sure you are using a modern browser (Chrome, Edge, Firefox).
- If there are errors related to FFmpeg, ensure you are running from within the cloned/unzipped repo—**ffmpeg is already included and requires no additional setup.**
- The app only supports Python 3.11.x. If you have issues, confirm your Python version.

Need Help?

Contact 1busyguy @imgMotion on X
