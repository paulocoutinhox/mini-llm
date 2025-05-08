import torch


def memory_less_64gb():
    """
    Check if the system has less than 64GB of memory.
    This is useful for determining when to enable memory-saving techniques
    like attention slicing for large models.

    Returns:
        bool: True if system memory is less than 64GB, False otherwise
    """
    try:
        import psutil

        mem_info = psutil.virtual_memory()
        total_gb = mem_info.total / (1024**3)  # Convert to GB
        print(f"🧠 System memory: {total_gb:.1f} GB")
        return total_gb < 64
    except ImportError:
        # If psutil is not available, assume memory is limited
        print("⚠️ Assuming limited memory, since psutil is not installed")
        return True


def get_device(optimize_memory=True):
    """
    Detect and return the best available device (GPU > MPS > CPU)
    with proper configuration for optimal performance

    Args:
        optimize_memory (bool): Whether to apply memory optimizations for Apple Silicon

    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🟩 Using CUDA GPU")

        # Set CUDA specific optimizations
        torch.backends.cudnn.benchmark = True

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("🟨 Using Apple MPS GPU")

        # Apply M1/M2/M3 specific optimizations
        if optimize_memory:
            # Limit memory usage for MPS
            torch.set_num_threads(6)  # Limit CPU threads for MPS

            # Empty cache to free up memory
            torch.mps.empty_cache()

            print("⚡ Applied memory optimizations for Apple Silicon")
    else:
        device = torch.device("cpu")
        print("🟥 Using CPU")

        # CPU specific optimizations
        if optimize_memory:
            torch.set_num_threads(8)  # Limit CPU threads

    return device


def get_device_info():
    """
    Get information about the current device including memory usage

    Returns:
        dict: Device information
    """
    info = {
        "device_type": None,
        "memory_allocated": None,
        "memory_reserved": None,
        "memory_total": None,
    }

    if torch.cuda.is_available():
        info["device_type"] = "cuda"
        info["memory_allocated"] = torch.cuda.memory_allocated() / 1024**3
        info["memory_reserved"] = torch.cuda.memory_reserved() / 1024**3
        info["memory_total"] = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        info["device_type"] = "mps"
        # MPS doesn't have built-in memory tracking like CUDA
        # These values will be None for MPS

    else:
        info["device_type"] = "cpu"
        # CPU memory tracking requires external libraries

    return info


def show_device_info():
    device_info = get_device_info()

    device_info = {
        "".join(word.capitalize() for word in key.split("_")): value
        for key, value in device_info.items()
    }

    print("\n📊 Device Information:")

    for key, value in device_info.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  - {key}: {value:.2f} GB")
            else:
                print(f"  - {key}: {value}")
        else:
            print(f"  - {key}: Not available")
    print("")
