import sys
import importlib
import logging
from pathlib import Path

# Configure runtime logging for clean terminal output
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def check_required_packages() -> bool:
    """
    Defensively validates the import integrity of the core tech stack.
    Returns True if all modules load successfully; False otherwise.
    """
    required_packages = ["torch", "cv2", "ultralytics", "mcp"]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            logging.error(f"Environmental Setup Failure: Core package '{package}' is not installed.")
            return False
            
    logging.info("Core software stack verified successfully.")
    return True

def check_cuda() -> bool:
    """
    Verifies if a CUDA-capable GPU is available for real-time inference acceleration.
    Logs warnings or failures depending on host execution environments.
    """
    import torch
    if not torch.cuda.is_available():
        logging.warning("Hardware Performance Warning: CUDA GPU is not accessible by PyTorch.")
        logging.warning("Real-time FPS benchmarks cannot be maintained on CPU execution hosts.")
        # Do not fail completely if only CUDA is missing - CPU can still run evaluation.
        return True
        
    logging.info(f"Hardware Optimization Verified: Target GPU identified -> {torch.cuda.get_device_name(0)}")
    return True

def check_directories() -> bool:
    """
    Verifies the absolute existence of crucial runtime workspace directories.
    Returns True if the required structure exists; False if structure is broken.
    """
    required_dirs = [
        Path("data/raw"),
        Path("data/sample_videos"),
        Path("benchmark"),
        Path("src")
    ]
    
    missing_detected = False
    for directory in required_dirs:
        if not directory.exists():
            logging.error(f"Workspace Integrity Failure: Missing required path -> '{directory}'")
            missing_detected = True
            
    if missing_detected:
        logging.error("Initialization halted. Please rebuild missing path trees before streaming.")
        return False
        
    logging.info("Workspace directory structure verified successfully.")
    return True

def main():
    print("====================================================")
    print("   RUNTIME ENVIRONMENT INITIALIZATION INITIALIZED   ")
    print("====================================================")
    
    # 1. Run package import validation
    if not check_required_packages():
        sys.exit("Execution Aborted: Missing software dependencies.")
        
    # 2. Run hardware accelerator checks
    if not check_cuda():
        sys.exit("Execution Aborted: Non-accelerated hardware detected.")
        
    # 3. Run path tree validation
    if not check_directories():
        sys.exit("Execution Aborted: Incomplete local workspace layout.")
        
    print("====================================================")
    print("   ALL SYSTEMS OPERATIONAL: COMPONENT VALIDATION PASSED")
    print("====================================================")

if __name__ == "__main__":
    main()
