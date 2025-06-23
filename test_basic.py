#!/usr/bin/env python3
"""
Basic functionality test for OpenInferencev2
Ensures core components work correctly after installation
"""
import sys
import os
import traceback
import time

def test_imports():
    """Test all core module imports"""
    print("\n=== Testing Core Imports ===")
    
    try:
        from openinferencev2.config import Config
        print("✓ Config imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Config: {e}")
        return False
        
    try:
        from openinferencev2.openinferencev2 import OpenInferencev2Engine, InferenceRequest, InferenceResponse
        print("✓ OpenInferencev2Engine imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import OpenInferencev2Engine: {e}")
        return False
        
    try:
        from openinferencev2.scheduler import RequestScheduler
        print("✓ RequestScheduler imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import RequestScheduler: {e}")
        return False
        
    try:
        from openinferencev2.monitor import PerformanceMonitor
        print("✓ PerformanceMonitor imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PerformanceMonitor: {e}")
        return False
        
    try:
        from openinferencev2.optimization import ModelOptimizer
        print("✓ ModelOptimizer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ModelOptimizer: {e}")
        return False
        
    return True

def test_config():
    """Test configuration management"""
    print("\n=== Testing Configuration ===")
    
    try:
        from openinferencev2.config import Config
        
        # Test default configuration
        config = Config()
        print("✓ Default configuration created")
        
        # Test configuration access
        assert config.num_gpus >= 1, "Invalid default GPU count"
        assert config.max_batch_size > 0, "Invalid batch size"
        print("✓ Configuration parameters accessible")
        
        # Test configuration update
        config.update({'test_param': 'test_value'})
        assert config.get('test_param') == 'test_value', "Configuration update failed"
        print("✓ Configuration update works")
        
        # Test validation
        errors = config.validate()
        if not errors:
            print("✓ Configuration validation passes")
        else:
            print(f"! Configuration validation warnings: {errors}")
            
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_data_structures():
    """Test core data structures"""
    print("\n=== Testing Data Structures ===")
    
    try:
        from openinferencev2.openinferencev2 import InferenceRequest, InferenceResponse
        
        # Test InferenceRequest
        request = InferenceRequest(
            id="test_request",
            prompt="Hello, world!",
            max_tokens=50,
            temperature=0.7
        )
        print("✓ InferenceRequest created successfully")
        
        # Test InferenceResponse
        response = InferenceResponse(
            id="test_request",
            text="Hello! How can I help you today?",
            tokens=[1, 2, 3, 4, 5],
            latency=0.123,
            tokens_per_second=40.5,
            finish_reason="length"
        )
        print("✓ InferenceResponse created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Data structures test failed: {e}")
        traceback.print_exc()
        return False

def test_performance_monitor():
    """Test performance monitoring"""
    print("\n=== Testing Performance Monitor ===")
    
    try:
        from openinferencev2.monitor import PerformanceMonitor
        
        # Create monitor
        monitor = PerformanceMonitor(history_size=100)
        print("✓ PerformanceMonitor created successfully")
        
        # Test stats
        stats = monitor.get_current_stats()
        assert isinstance(stats, dict), "Stats should be a dictionary"
        print("✓ Performance statistics accessible")
        
        # Test health check
        health = monitor.health_check()
        print("✓ Health check works")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance monitor test failed: {e}")
        traceback.print_exc()
        return False

def test_engine_creation():
    """Test engine creation (without model loading)"""
    print("\n=== Testing Engine Creation ===")
    
    try:
        from openinferencev2.openinferencev2 import OpenInferencev2Engine
        from openinferencev2.config import Config
        
        # Create mock model path
        model_path = "/tmp/mock_model"
        
        # Create config
        config = Config({
            'num_gpus': 1,
            'max_batch_size': 4,
            'use_fp16': False  # Safer for testing
        })
        
        # Create engine (don't load model)
        engine = OpenInferencev2Engine(model_path, config.__dict__)
        print("✓ OpenInferencev2Engine creation works")
        
        # Test stats
        stats = engine.get_performance_stats()
        assert isinstance(stats, dict), "Stats should be a dictionary"
        print("✓ Engine performance stats accessible")
        
        return True
        
    except Exception as e:
        print(f"✗ Engine creation test failed: {e}")
        traceback.print_exc()
        return False

def test_optimizer():
    """Test model optimizer"""
    print("\n=== Testing Model Optimizer ===")
    
    try:
        from openinferencev2.optimization import ModelOptimizer
        from openinferencev2.openinferencev2 import OpenInferencev2Engine
        from openinferencev2.config import Config
        
        # Create mock engine
        config = Config()
        engine = OpenInferencev2Engine("/tmp/mock", config.__dict__)
        
        # Create optimizer
        optimizer = ModelOptimizer(engine)
        print("✓ ModelOptimizer created successfully")
        
        # Test status
        status = optimizer.get_optimization_status()
        assert isinstance(status, dict), "Status should be a dictionary"
        print("✓ Optimization status accessible")
        
        return True
        
    except Exception as e:
        print(f"✗ Optimizer test failed: {e}")
        traceback.print_exc()
        return False

def test_scheduler():
    """Test request scheduler"""
    print("\n=== Testing Request Scheduler ===")
    
    try:
        from openinferencev2.scheduler import RequestScheduler
        from openinferencev2.openinferencev2 import OpenInferencev2Engine
        from openinferencev2.config import Config
        
        # Create mock engine
        config = Config()
        engine = OpenInferencev2Engine("/tmp/mock", config.__dict__)
        
        # Create scheduler
        scheduler = RequestScheduler(engine, max_batch_size=4, max_queue_size=100)
        print("✓ RequestScheduler created successfully")
        
        # Test stats
        stats = scheduler.get_stats()
        assert isinstance(stats, dict), "Stats should be a dictionary"
        print("✓ Scheduler statistics accessible")
        
        return True
        
    except Exception as e:
        print(f"✗ Scheduler test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("OpenInferencev2 Basic Functionality Test")
    print("=" * 50)
    
    test_functions = [
        test_imports,
        test_config,
        test_data_structures,
        test_performance_monitor,
        test_engine_creation,
        test_optimizer,
        test_scheduler
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_func in test_functions:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            traceback.print_exc()
    
    print(f"\n{'=' * 50}")
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("All tests passed!")
        return 0
    else:
        print(f"{total - passed} tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 