"""Identity control; retain parent checkpoint byte-for-byte. Not a weight repair."""
from pathlib import Path
import shutil
def retain(parent, output):
    shutil.copyfile(parent, output)
if __name__ == "__main__":
    import sys
    retain(sys.argv[1], sys.argv[2])
