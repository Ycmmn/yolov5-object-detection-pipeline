# YOLOv5 Object Detection Pipeline – Full Documentation

## (1) Importing Libraries

With one line of code, all the required libraries can be installed in a Jupyter cell:

```bash
pip install torch torchvision opencv-python numpy
```

- In your code you used **torch** and **cv2 (opencv-python)** → these must be installed.  
- **numpy** and **torchvision** aren’t directly imported in your code, but YOLOv5 uses them internally → better to put them in `requirements.txt`.

So the list of required libraries is:

```
torch torchvision opencv-python numpy
```

With just these 4 in `requirements.txt`, everything works fine.

---

### Why not everything?

Not everything you `import` must go into `requirements.txt`.  
Only external libraries (installed via pip) go there.

- `import sys, time, collections` → all part of Python’s standard library, installed automatically → **not needed** in requirements.  
- `import cv2, torch` → external libraries, must go into requirements:
  - `opencv-python` for cv2  
  - `torch` (and better add `torchvision`)  

---

## Libraries and their roles

### sys
- For accessing command-line arguments.  
  Example:
  ```bash
  python detect.py video.mp4
  ```
  With `sys.argv[1]` you can read `video.mp4`.

### time
- For working with time.  
- Used to measure execution speed and FPS.  
  ```python
  start = time.time()
  ...
  end = time.time()
  duration = end - start
  ```

### collections
- `Counter` is a helper from `collections`.  
- It counts repeated items easily.  
- In YOLO: each frame outputs labels like `"person", "car", "dog"`.  
- `Counter` tells you how many persons, cars, dogs, etc.

### cv2 (OpenCV)
- Image and video processing library.  
- Uses:
  - Read & display video (`cv2.VideoCapture`, `cv2.imshow`)  
  - Draw bounding boxes, labels  
  - Save output video  

### torch (PyTorch)
- For deep learning models.  
- Here used for YOLOv5:
  - Load pretrained model  
  - Run object detection on frames  

Although you don’t `import torchvision` or `numpy` explicitly, YOLOv5 uses them internally → best to include.

---

## Special lines in your code

```python
cv2.setUseOptimized(True)
```
- Tells OpenCV to use optimized internal algorithms → faster execution.

```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```
- Chooses execution device: GPU if available, otherwise CPU.  
- Makes the code portable across systems.

```python
torch.backends.cudnn.benchmark = True
```
- Lets PyTorch test multiple cuDNN algorithms for fixed input size and pick the fastest.  
- More stable FPS and faster inference.

---

## The `WANT` set

```python
WANT = {'person','car','truck','bus','motorcycle','bicycle','dog','cat','bird','horse','cow','sheep'}
```

- A Python **set** ensures:
  - No duplicates  
  - Very fast membership checking (`if label in WANT`)  

**Usage here**:  
- YOLO detects many classes.  
- You only care about a subset → `WANT` filters the predictions.

---

## (2) Loading & Preparing the Model

### Function

```python
def load_model():
```
Loads a pretrained YOLOv5 model and returns it.

### Loading

```python
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(DEVICE)
```
- Downloads YOLOv5 from Ultralytics repo.  
- `'yolov5s'` = small version (fast, lightweight).  
- `pretrained=True` → loads COCO pretrained weights.  
- `.to(DEVICE)` → moves model to GPU if available.

### Thresholds
- `model.conf = 0.35` → discard boxes below 35% confidence.  
- `model.iou = 0.45` → remove overlapping boxes above 45% IoU.  
- `model.max_det = 300` → max 300 boxes per image.  

### FP16
```python
if DEVICE == 'cuda':     
    model.half()
```
- On GPU: converts model to float16 for speed and memory efficiency.  
- On CPU: not supported.  

### Classes
```python
model.names
```
- Contains all class names (dict or list).  
- With `WANT`, filter to only specific classes.  
- Use `model.classes = [...]` so YOLO outputs only desired ones.

---

## (3) Opening Video Source

```python
def open_source(src):
```
- `src=0` → default webcam.  
- `"video.mp4"` → video file.  
- `"rtsp://..."` / `"http://..."` → network stream.  

Steps:
1. `cap = cv2.VideoCapture(src)` → open source.  
2. `if not cap.isOpened(): raise SystemExit("❌")` → check it’s valid.  
3. `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` → minimal buffer → less lag.  
4. Return `cap` object.  

---

## (4) Warmup Model

```python
def warmup(model):
```
- Creates a dummy tensor (1×3×640×640).  
- Runs one forward pass.  
- Purpose: remove initial lag (GPU kernel setup).  
- Uses `torch.no_grad()` since gradients aren’t needed.

---

## (5) Get Frame

```python
def get_frame(cap):
    ret, frame = cap.read()
    return frame if ret else None
```
- Reads one frame from source.  
- Returns frame as NumPy array if successful, else `None`.

---

## (6) Preprocessing

```python
def preprocess(frame):
    return frame
```
- Placeholder function.  
- Not needed for YOLOv5 (it handles preprocessing).  
- But useful if switching to a different model later.

---

## (7) Inference

```python
@torch.no_grad()
def infer(model, image, size=640):
    model.eval()
    results = model(image, size=size)
    return results
```
- Runs the model.  
- `model.eval()` → inference mode (BatchNorm/Dropout behave correctly).  
- `torch.no_grad()` → disables gradient calculation (faster, less memory).  

---

## (8) Postprocessing

```python
def postprocess(results, names):
```
- Extracts predictions.  
- Converts class IDs to names using `names`.  
- Returns `labels` and a `Counter(labels)`.

**Example:**
```python
['person', 'car'], {'person': 1, 'car': 1}
```

---

## (9) Drawing & Status Bar

```python
def draw_and_compose(result, counts, fps):
```
- Draws YOLO boxes (`result.render()`).  
- Adds FPS and counts text at the top (with shadow for readability).  
- Returns final frame.

---

## (10) Video Saving

```python
def maybe_open_writer(save_flag, writer, frame_like, out_path='output.mp4', fps=30):
```
- Creates a `cv2.VideoWriter` if needed.  
- Ensures it’s only created once.  

```python
def write_and_show(writer, vis):
```
- If writer exists → saves current frame.  
- Always shows frame with `cv2.imshow`.

---

## (11) Cleanup

```python
def cleanup(cap, writer):
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
```
- Releases video source and writer.  
- Closes all OpenCV windows.

---

## Runtime Notes
- `t_prev = time.time()` → store initial time.  
- Main loop in `try`/`finally` → ensures `cleanup()` always runs.
