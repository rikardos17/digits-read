# Digits reader

This is a simple digit reader using easyocr, to read the digits in a video stream.


# Install

Install the dependencies, in this case I'm using Ubuntu 20.04

Clone the repository

```bash
git clone https://github.com/DavidABSiepmann/digits-read.git
```

**Attention:** the `easyocr` broke the opencv display, the function `cv2.imshow()` is not working with easyocr, so create a virtual environment to run the script.

Create a virtual environment and install the requirements

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requeriments.txt
```

# Run

To run the script, you need to pass the video path, if the roi is not passed, the script will ask you to select the roi in the video, and the position of the digits in the image. Exemple without roi:

```bash
python3 read_digits.py -v <video_path>
```

The other parameters are:

```bash
-v, --video: path to the video
-p, --project: path to the project folder
-n, --name: name of the experiment
-d, --draw: draw the results in a video 
-r, --roi: roi coordinates
-w, --wfit: size of the window to fit the polynomial correction
-c, --confidence: confidence threshold of the ocr detection to consider a digit
-m, --mm: moving average threshold to consider a digit
-pd, --polydegree: polynomial degree to fit the correction
-opim, --offsetprocessed: offset of the processed frame to draw inside the original frame
-oplt, --offsetplot: offset of the plot to draw inside the original frame
-fs, --figsize: figure size in matplotlib, used only if draw is true
```

Exemple drawing the results in a video:

```bash
python read_digits.py -v input_source.mp4 -d -r 849 297 140 89 -opim 680 650 -oplt 1219 0 -fs 7 6
```

Exemple without drawing the results in a video:

```bash
python read_digits.py -v input_source.mp4 -r 849 297 140 89
```