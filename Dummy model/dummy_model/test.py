import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
print("PyTorch CUDA version:", torch.version.cuda)
print("Torch build:", torch.__version__)


import huggingface_hub, transformers, datasets
print("hf_hub:", huggingface_hub.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)