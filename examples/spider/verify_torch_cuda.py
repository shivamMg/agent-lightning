import subprocess
import torch

print("Torch version:", torch.__version__)
print("Built with CUDA:", torch.version.cuda)
print("Built with ROCm:", torch.version.hip)

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("CUDA device capability:", torch.cuda.get_device_capability())
    print("BF16 supported:", torch.cuda.is_bf16_supported())
    print("TF32 supported:", torch.cuda.is_tf32_supported())

    print("\nActual GPU memory usage (nvidia-smi):")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    print(f"{'GPU':<4} {'Name':<30} {'Used':<10} {'Total'}")
    for line in result.stdout.strip().split("\n"):
        idx, name, used, total = [x.strip() for x in line.split(",")]
        print(f"{idx:<4} {name:<30} {used}MiB    {total}MiB")

    print("\nCurrent python process GPU memory usage (torch):")
    print(f"{'GPU':<4} {'Name':<30} {'Memory Allocated/Reserved/Total'}")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        allocated = torch.cuda.memory_allocated(i) // (1024 ** 2)
        reserved = torch.cuda.memory_reserved(i) // (1024 ** 2)
        total = torch.cuda.get_device_properties(i).total_memory // (1024 ** 2)
        print(f"{i:<4} {name:<30} {allocated}MiB / {reserved}MiB / {total}MiB")
