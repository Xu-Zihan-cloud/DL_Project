import os
import torch
import multiprocessing

def optimize_cpu_performance(threads_per_process=32):
    """
    Optimizes PyTorch for high-core count CPU execution (e.g., AMD EPYC).
    Prevents thread over-subscription and enables hardware acceleration.
    """
    # Set thread limits for major BLAS/Parallel libraries
    os.environ["OMP_NUM_THREADS"] = str(threads_per_process)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_process)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_process)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads_per_process)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads_per_process)
    
    # Configure PyTorch threading
    torch.set_num_threads(threads_per_process)
    torch.set_num_interop_threads(threads_per_process)
    
    # Enable MKLDNN
    torch.backends.mkldnn.enabled = True
    
    print(f"CPU Optimization: Threads limited to {threads_per_process}")
    print(f"MKLDNN: {torch.backends.mkldnn.enabled}")

def get_autocast_context():
    """
    Returns a BFloat16 autocast context for AMD EPYC AVX-512 BF16 support.
    """
    # Note: device_type='cpu' for CPU autocast
    return torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
