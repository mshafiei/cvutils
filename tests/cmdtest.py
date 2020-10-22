import numpy as np
import argparse
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Deploying command')

    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    print(args.command)