#%%
from ultralytics import YOLO


model = YOLO('yolov8n.pt')


results = model(source='test.mp4', show=False, conf=0.05, save=True, classes=2, save_conf=True, line_width=1)