import sys
from pathlib import Path

# Add core folder to sys.path if not already
core_path = Path(__file__).parent.parent / "core"
if str(core_path) not in sys.path:
    sys.path.insert(0, str(core_path))

import detect  # Import your detect.py module from core

def main():
    # Start the pipeline with detect
    detect.main()  # Assuming detect.py has a main() function

if __name__ == "__main__":
    main()