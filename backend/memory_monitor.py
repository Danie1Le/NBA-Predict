#!/usr/bin/env python3
"""
Memory monitoring utility for NBA Predict API
Helps prevent memory exceeded errors
"""

import psutil
import os
import gc

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

def log_memory_usage(context=""):
    """Log current memory usage"""
    memory_mb = get_memory_usage()
    print(f"💾 Memory usage {context}: {memory_mb:.1f} MB")
    return memory_mb

def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()
    print("🧹 Memory cleanup performed")

def check_memory_limit(limit_mb=450):
    """Check if memory usage is approaching limit"""
    memory_mb = get_memory_usage()
    if memory_mb > limit_mb:
        print(f"⚠️ Memory usage high: {memory_mb:.1f} MB (limit: {limit_mb} MB)")
        cleanup_memory()
        return True
    return False

def memory_efficient_model_training():
    """Context manager for memory-efficient model training"""
    class MemoryContext:
        def __enter__(self):
            log_memory_usage("before training")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            cleanup_memory()
            log_memory_usage("after training")
    
    return MemoryContext()
