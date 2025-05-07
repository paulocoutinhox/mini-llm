import torch


def get_device():
    """
    Detect and return the best available device (GPU > MPS > CPU)
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🟩 Using CUDA GPU")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("🟨 Using Apple MPS GPU")
    else:
        device = torch.device("cpu")
        print("🟥 Using CPU")

    return device
