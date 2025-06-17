#!/usr/bin/env python3
"""
Test script to demonstrate logging and performance monitoring in ParkingSpace system.
This script shows you where logs are saved and how to view performance data.
"""

import os
import sys
import time
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_logging():
    """Test logging functionality and show where logs are saved"""
    print("=== LOGGING TEST ===")
    
    from parkingspace.logger import setup_logging, get_logger
    
    # Test 1: Console logging only (default)
    print("\n1. Console Logging (default behavior):")
    logger = setup_logging()
    logger.info("This is an INFO message - appears in console")
    logger.warning("This is a WARNING message - appears in console")
    logger.error("This is an ERROR message - appears in console")
    
    # Test 2: File logging
    print("\n2. File Logging:")
    log_file = "parkingspace.log"
    logger_with_file = setup_logging(log_file=log_file)
    logger_with_file.info("This message is saved to both console and file")
    logger_with_file.warning("Performance data will also be logged here")
    
    if os.path.exists(log_file):
        print(f"✅ Log file created: {os.path.abspath(log_file)}")
        print("📄 Log file contents:")
        with open(log_file, 'r') as f:
            content = f.read()
            print(content)
    else:
        print("❌ Log file was not created")
    
    return log_file

def test_performance_monitoring():
    """Test performance monitoring and show how to view performance data"""
    print("\n=== PERFORMANCE MONITORING TEST ===")
    
    from parkingspace.performance import PerformanceMonitor, get_performance_monitor
    from parkingspace.logger import get_logger
    
    logger = get_logger()
    
    # Create performance monitor
    monitor = PerformanceMonitor()
    
    print("\n1. Simulating frame processing...")
    
    # Simulate processing 5 frames
    for i in range(5):
        print(f"Processing frame {i+1}...")
        
        # Start timing
        monitor.start_frame_processing()
        
        # Simulate some work (detection time)
        detection_start = time.time()
        time.sleep(0.1)  # Simulate detection work
        detection_time = time.time() - detection_start
        
        # Simulate more processing
        time.sleep(0.05)  # Simulate other processing
        
        # End timing and get metrics
        metrics = monitor.end_frame_processing(detection_time)
        
        print(f"  Frame {i+1} metrics:")
        print(f"    Total processing time: {metrics.frame_processing_time:.3f}s")
        print(f"    Detection time: {metrics.detection_time:.3f}s")
        print(f"    Memory usage: {metrics.memory_usage:.1f}%")
        print(f"    CPU usage: {metrics.cpu_usage:.1f}%")
        if metrics.gpu_memory is not None:
            print(f"    GPU memory: {metrics.gpu_memory:.1f}%")
    
    print("\n2. Performance Report:")
    monitor.log_performance_report(last_n=5)
    
    print("\n3. Average Performance Metrics:")
    avg_metrics = monitor.get_average_metrics(last_n=5)
    if avg_metrics:
        fps = 1.0 / avg_metrics.frame_processing_time if avg_metrics.frame_processing_time > 0 else 0
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Average processing time: {avg_metrics.frame_processing_time:.3f}s")
        print(f"  Average detection time: {avg_metrics.detection_time:.3f}s")
        print(f"  Average memory usage: {avg_metrics.memory_usage:.1f}%")
        print(f"  Average CPU usage: {avg_metrics.cpu_usage:.1f}%")

def test_application_logging():
    """Test how the main application uses logging"""
    print("\n=== APPLICATION LOGGING TEST ===")
    
    from parkingspace.config import get_config, reset_config
    from parkingspace.logger import setup_logging
    
    # Reset config to start fresh
    reset_config()
    
    # Setup logging with file
    log_file = "app_test.log"
    logger = setup_logging(log_file=log_file)
    
    print(f"\n1. Testing configuration loading with logging to: {os.path.abspath(log_file)}")
    
    # Test config loading (this will generate log messages)
    config = get_config()
    logger.info(f"Device detected: {config.device}")
    logger.info(f"Model path: {config.model_path}")
    logger.info(f"Performance logging interval: {config.performance.log_interval} frames")
    
    if os.path.exists(log_file):
        print(f"✅ Application log file created: {os.path.abspath(log_file)}")
        print("📄 Application log contents:")
        with open(log_file, 'r') as f:
            content = f.read()
            print(content)

def main():
    """Main test function"""
    print("🎯 ParkingSpace Logging and Performance Monitoring Demo")
    print("=" * 60)
    
    # Test logging
    log_file = test_logging()
    
    # Test performance monitoring
    test_performance_monitoring()
    
    # Test application logging
    test_application_logging()
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY - Where to find logs and performance data:")
    print("=" * 60)
    
    print("\n🔍 LOGGING:")
    print("  • By default: Logs appear in CONSOLE only")
    print("  • With log file: Logs appear in CONSOLE + specified file")
    print("  • Current working directory:", os.getcwd())
    print("  • Log files created in this demo:")
    
    for log_name in ["parkingspace.log", "app_test.log"]:
        if os.path.exists(log_name):
            print(f"    - {os.path.abspath(log_name)}")
    
    print("\n📈 PERFORMANCE MONITORING:")
    print("  • Performance metrics are logged to the same logger")
    print("  • Metrics include: FPS, processing time, CPU/memory usage")
    print("  • Performance reports are logged every N frames (configurable)")
    print("  • Use monitor.log_performance_report() to see performance data")
    print("  • Performance data is stored in memory (PerformanceMonitor.metrics_history)")
    
    print("\n🚀 TO USE IN YOUR APPLICATION:")
    print("  1. Import: from parkingspace.logger import setup_logging")
    print("  2. Setup with file: logger = setup_logging(log_file='my_app.log')")
    print("  3. Use logger: logger.info('Your message here')")
    print("  4. Performance: from parkingspace.performance import get_performance_monitor")
    
    # Cleanup
    for log_file in ["parkingspace.log", "app_test.log"]:
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
                print(f"\n🧹 Cleaned up: {log_file}")
            except:
                pass

if __name__ == "__main__":
    main()
