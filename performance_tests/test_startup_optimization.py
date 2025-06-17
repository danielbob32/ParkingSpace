#!/usr/bin/env python3
"""
Startup time optimization test - compares normal vs fast startup modes.
"""

import time
import sys
import os
import subprocess
from pathlib import Path

def test_import_time():
    """Test just import time"""
    print("🔍 Testing Import Time")
    print("-" * 30)
    
    start_time = time.time()
    import src.parkingspace
    import_time = time.time() - start_time
    
    print(f"📦 Import time: {import_time:.3f}s")
    return import_time

def test_normal_startup():
    """Test normal startup time"""
    print("\n🐌 Testing Normal Startup")
    print("-" * 30)
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    start_time = time.time()
    
    # Import and run with fast_mode=False
    from src.parkingspace.main import main
    from src.parkingspace.logger import setup_logging
    
    logger = setup_logging()
    
    try:
        # Simulate normal startup (without video processing)
        from src.parkingspace.config import get_config
        from src.parkingspace.capabilities import get_capability_detector
        from src.parkingspace.services import ParkingSpaceService
        
        config = get_config()
        detector = get_capability_detector()
        capabilities = detector.detect_system_capabilities()
        
        parking_service = ParkingSpaceService(config)
        parking_service.model_service.load_model()
        parking_service.region_service.load_regions()
        
        startup_time = time.time() - start_time
        print(f"⏱️  Normal startup time: {startup_time:.3f}s")
        return startup_time
        
    except Exception as e:
        print(f"❌ Normal startup failed: {e}")
        return None

def test_fast_startup():
    """Test fast startup time"""
    print("\n🚀 Testing Fast Startup")
    print("-" * 30)
    
    start_time = time.time()
    
    try:
        # Import fast startup modules
        from src.parkingspace.fast_startup import get_fast_startup_manager
        from src.parkingspace.config import get_config
        from src.parkingspace.services import ParkingSpaceService
        
        # Initialize fast startup
        fast_startup = get_fast_startup_manager()
        fast_startup.optimize_imports()
        
        # Start parallel operations
        fast_startup.parallel_capability_detection()
        
        # Load config
        config = get_config()
        
        # Start model loading in parallel
        fast_startup.parallel_model_loading(config)
        
        # Get capabilities
        capabilities = fast_startup.parallel_initializer.get_result("capability_detection")
        
        # Initialize service
        parking_service = ParkingSpaceService(config)
        
        # Try to use pre-loaded model
        model = fast_startup.parallel_initializer.get_result("model_loading", timeout=5.0)
        if model:
            parking_service.model_service.model = model
            parking_service.model_service._warmup_done = True
        else:
            parking_service.model_service.load_model()
        
        parking_service.region_service.load_regions()
        
        startup_time = time.time() - start_time
        print(f"⚡ Fast startup time: {startup_time:.3f}s")
        
        fast_startup.log_startup_summary()
        return startup_time
        
    except Exception as e:
        print(f"❌ Fast startup failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_lazy_import_benefits():
    """Test benefits of lazy importing"""
    print("\n📦 Testing Lazy Import Benefits")
    print("-" * 30)
    
    from src.parkingspace.fast_startup import LazyImporter
    
    lazy_importer = LazyImporter()
    
    # Test lazy importing heavy modules
    heavy_modules = ['torch', 'ultralytics', 'cv2', 'numpy']
    
    for module in heavy_modules:
        start_time = time.time()
        result = lazy_importer.lazy_import(module)
        import_time = time.time() - start_time
        
        status = "✅" if result else "❌"
        print(f"  {status} {module}: {import_time:.3f}s")
    
    # Show summary
    summary = lazy_importer.get_import_summary()
    total_time = sum(summary.values())
    print(f"\n📊 Total lazy import time: {total_time:.3f}s")

def compare_startup_methods():
    """Compare different startup methods"""
    print("🏁 Startup Time Comparison")
    print("=" * 50)
    
    results = {}
    
    # Test 1: Import time only
    results['import'] = test_import_time()
    
    # Test 2: Lazy import benefits
    test_lazy_import_benefits()
    
    # Test 3: Normal startup
    results['normal'] = test_normal_startup()
    
    # Test 4: Fast startup
    results['fast'] = test_fast_startup()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 STARTUP TIME COMPARISON SUMMARY")
    print("=" * 50)
    
    if results['import']:
        print(f"📦 Import only: {results['import']:.3f}s")
    
    if results['normal']:
        print(f"🐌 Normal startup: {results['normal']:.3f}s")
    
    if results['fast']:
        print(f"🚀 Fast startup: {results['fast']:.3f}s")
        
        if results['normal']:
            improvement = results['normal'] - results['fast']
            percentage = (improvement / results['normal']) * 100
            print(f"⚡ Improvement: {improvement:.3f}s ({percentage:.1f}% faster)")
    
    print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
    
    if results.get('fast') and results.get('normal'):
        if results['fast'] < results['normal']:
            print("   ✅ Fast startup is working - use fast mode for better performance")
        else:
            print("   ⚠️  Fast startup may not be optimal for your system")
    
    if results.get('import', 0) > 2.0:
        print("   🔧 Heavy import time detected - consider lazy loading more modules")
    
    if results.get('normal', 0) > 10.0:
        print("   📉 Slow startup detected - fast mode highly recommended")
    elif results.get('normal', 0) > 5.0:
        print("   🔧 Moderate startup time - fast mode recommended")
    else:
        print("   ✅ Startup time is already good")

def create_startup_optimization_script():
    """Create a script to enable fast startup by default"""
    script_content = '''#!/usr/bin/env python3
"""
Optimized ParkingSpace Detection launcher with fast startup enabled.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.parkingspace.main import main

if __name__ == "__main__":
    # Parse command line arguments
    config_file = None
    video_file = None
    
    for arg in sys.argv[1:]:
        if arg.endswith('.json'):
            config_file = arg
        elif arg.endswith(('.mp4', '.avi', '.mov')):
            video_file = arg
    
    # Run with fast startup enabled
    main(config_file, video_file, fast_mode=True)
'''
    
    with open('run_fast.py', 'w') as f:
        f.write(script_content)
    
    print("\n🚀 Created optimized launcher: run_fast.py")
    print("   Usage: python run_fast.py [config.json] [video.mp4]")

def main():
    """Main function"""
    print("⚡ ParkingSpace Startup Time Optimization Test")
    print("🎯 Testing different startup optimization strategies...")
    print()
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    try:
        compare_startup_methods()
        create_startup_optimization_script()
        
        print(f"\n🏆 Startup optimization analysis completed!")
        print("💡 Use the fast startup mode for better performance")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
