"""Benchmark detector + aligner + recognizer candidates on this exact GPU."""
import time, os, sys
import numpy as np, cv2, onnxruntime as ort

MODELS = "/home/inomjon/projectAI/face_rec/face_recognition_airi/models"
PROBE  = "/tmp/claude-1000/-home-inomjon-projectAI-face-rec-face-recognition-airi/5ed61b02-fc69-491a-b559-68baef899b6f/scratchpad/probe"
CUDA = [("CUDAExecutionProvider", {"device_id":0}), "CPUExecutionProvider"]

def timeit(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    t=time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter()-t)/iters*1000

def sess(path, providers=CUDA):
    o = ort.SessionOptions(); o.log_severity_level = 3
    return ort.InferenceSession(os.path.join(MODELS,path), sess_options=o, providers=providers)

print("="*78); print("1. YuNet detector (OpenCV FaceDetectorYN)"); print("="*78)
frame4k = cv2.imread(f"{PROBE}/snap_in.jpg")
print(f"source frame: {frame4k.shape}")
for w,h in [(640,360),(960,544),(1280,736),(1920,1088)]:
    img = cv2.resize(frame4k,(w,h))
    det = cv2.FaceDetectorYN.create(f"{MODELS}/face_detection_yunet_2023mar.onnx","",(w,h),0.6,0.3,5000)
    ms = timeit(lambda: det.detect(img), 3, 20)
    print(f"  YuNet @ {w}x{h}: {ms:6.2f} ms   ({1000/ms:5.1f} fps)")

print()
print("="*78); print("2. YOLOv8n-face (ultralytics, CUDA)"); print("="*78)
try:
    from ultralytics import YOLO
    import torch, logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    m = YOLO(f"{MODELS}/yolov8n-face.pt"); m.to("cuda")
    for size in [640, 960, 1280]:
        img = cv2.resize(frame4k,(size, int(size*9/16)//32*32))
        ms = timeit(lambda: m.predict(img, verbose=False, device=0, imgsz=size), 5, 20)
        print(f"  YOLOv8n-face @ imgsz={size}: {ms:6.2f} ms   ({1000/ms:5.1f} fps)")
except Exception as e:
    print("  FAILED:", type(e).__name__, e)

print()
print("="*78); print("3. DFA mobilenet aligner (ONNX CUDA)"); print("="*78)
al = sess("dfa_mobilenet_aligner.onnx")
print("  providers:", al.get_providers()[:1])
for bs in [1,4,8,16]:
    x = np.random.randn(bs,3,224,224).astype(np.float32)
    ms = timeit(lambda: al.run(None,{"image":x}))
    print(f"  aligner bs={bs:2d} (224x224 in): {ms:6.2f} ms  -> {ms/bs:5.2f} ms/face")

print()
print("="*78); print("4. AdaFace recognizers (ONNX CUDA)"); print("="*78)
for name in ["adaface_ir101_webface12m.onnx","adaface_ir101_webface12m_fp16.onnx","adaface_ir18_webface4m.onnx"]:
    try:
        s = sess(name); prov = s.get_providers()[0].replace("ExecutionProvider","")
        inp = s.get_inputs()[0]; dt = np.float16 if "float16" in inp.type else np.float32
        row=[]
        for bs in [1,4,8,16]:
            x = np.random.randn(bs,3,112,112).astype(dt)
            ms = timeit(lambda: s.run(None,{inp.name:x}))
            row.append(f"bs{bs}={ms:6.2f}ms({ms/bs:5.2f}/f)")
        print(f"  {name:38s} [{prov}] {' '.join(row)}")
    except Exception as e:
        print(f"  {name:38s} FAILED: {type(e).__name__}: {str(e)[:90]}")
