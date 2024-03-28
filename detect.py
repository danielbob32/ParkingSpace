#%%
from ultralytics import YOLO


model = YOLO('yolov8x.pt')


results = model(source='Assets/Baseline Images/1.jpg', show=False, conf=0.05, save=True, classes=2, save_txt=True, save_conf=True, line_width=1)

