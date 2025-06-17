# 📊 ParkingSpace Detection - Logging & Performance Guide

## 🔍 Where are logs being saved?

### Current Behavior:
- **By default**: Logs are displayed in the **CONSOLE ONLY** (terminal/command prompt)
- **No file logging**: The application currently doesn't save logs to files automatically

### What logs show you:
```
2025-06-17 19:53:35,790 - parkingspace - INFO - Starting ParkingSpace Detection System
2025-06-17 19:53:35,810 - parkingspace - INFO - Using device: cuda
2025-06-17 19:54:04,670 - parkingspace - INFO - ParkingSpace Detection System completed successfully
```

### Log Information Includes:
1. **System startup/shutdown messages**
2. **Device detection** (CUDA/CPU)
3. **Configuration loading**
4. **Error messages** if something goes wrong
5. **Frame processing statistics** (every 10 frames by default)
6. **Final performance reports**

---

## 📈 Performance Monitoring - Where to see it?

### 🎯 Performance Data Locations:

#### 1. **Console Logs** (Real-time during processing)
Every 10 frames, you'll see logs like:
```
INFO - Frame 10: Processing time: 0.166s, Detection time: 0.104s, Empty spaces: 15, Vehicles detected: 23
INFO - Frame 20: Processing time: 0.158s, Detection time: 0.098s, Empty spaces: 12, Vehicles detected: 25
```

#### 2. **Final Performance Report** (At the end)
```
INFO - Performance Report (last 100 frames):
INFO -   Average frame processing time: 0.166 seconds
INFO -   Average detection time: 0.104 seconds  
INFO -   Average memory usage: 46.3%
INFO -   Average CPU usage: 29.9%
INFO -   Average GPU memory usage: 78.5%
INFO -   Estimated FPS: 6.0
```

### 📊 Performance Metrics Include:
- **Frame Processing Time** - Total time to process each frame
- **Detection Time** - Time spent on YOLO object detection
- **Memory Usage** - System RAM usage percentage
- **CPU Usage** - Processor usage percentage
- **GPU Memory Usage** - VRAM usage percentage (if using CUDA)
- **Estimated FPS** - Frames per second performance

---

## 🚀 How to Enable File Logging

### Option 1: Modify the code
In `src/parkingspace/main.py`, change:
```python
# Current (console only)
logger = setup_logging()

# Change to (console + file)
logger = setup_logging(log_file="parkingspace.log")
```

### Option 2: Use the demo script
I created `demo_with_logging.py` for you:
```bash
python demo_with_logging.py
```
This will:
- Run the detection system
- Save all logs to `parkingspace_run.log`
- Show you the log contents when finished

### Option 3: Use the new VS Code task
- Open Command Palette (`Ctrl+Shift+P`)
- Type "Tasks: Run Task"
- Select "📋 Run with File Logging"

---

## 🔧 Configuration Options

### Performance Logging Frequency
In `src/parkingspace/config.py`:
```python
@dataclass
class PerformanceConfig:
    log_interval: int = 10  # Log performance every N frames
```

Change `log_interval` to:
- `1` = Log every frame (verbose)
- `5` = Log every 5 frames
- `30` = Log every 30 frames (less verbose)

### Log Level
You can change the logging detail level:
```python
logger = setup_logging(level=logging.DEBUG)  # Very detailed
logger = setup_logging(level=logging.INFO)   # Standard (default)
logger = setup_logging(level=logging.WARNING) # Only warnings/errors
```

---

## 🛠️ Quick Commands to Test

### Test Current Logging:
```bash
python -m src.parkingspace.main
```

### Test with File Logging:
```bash
python demo_with_logging.py
```

### Test Performance Monitoring Only:
```bash
python test_logging_and_performance.py
```

### View Log Files:
```bash
# Windows
notepad parkingspace_run.log

# Or open in VS Code
code parkingspace_run.log
```

---

## 💡 Key Points

### Logs Tell You:
✅ **System status** (startup, shutdown, errors)  
✅ **Performance metrics** (FPS, processing times)  
✅ **Detection results** (cars found, empty spaces)  
✅ **Hardware usage** (CPU, memory, GPU)  

### Performance Data Shows:
✅ **Real-time processing speed**  
✅ **System resource usage**  
✅ **Detection accuracy timing**  
✅ **Overall system efficiency**  

### Current Default Behavior:
⚠️ **Console only** - no files saved automatically  
⚠️ **Performance logged every 10 frames**  
⚠️ **Final report shown at the end**  

---

## 🎯 Recommendations

1. **For Development**: Use `demo_with_logging.py` to save logs to files
2. **For Production**: Modify `main.py` to always log to files
3. **For Debugging**: Set `log_interval = 1` for detailed frame-by-frame data
4. **For Performance Analysis**: Check the final performance report for optimization

The performance monitoring system is quite comprehensive and will help you understand how well your parking space detection is performing!
