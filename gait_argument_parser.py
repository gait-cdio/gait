import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform gait analysis')
    parser.add_argument('--filename', required=True)
    parser.add_argument('--cached', default=False)
    parser.add_argument('--numOfTrackers', type=int, default=1)
    args = parser.parse_args()

    return args

