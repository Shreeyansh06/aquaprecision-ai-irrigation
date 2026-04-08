import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import run_inference

def main():
    run_inference()

if __name__ == "__main__":
    main()