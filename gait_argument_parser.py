import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform gait analysis')
    parser.add_argument('--filename', required=True)
    parser.add_argument('--cached', default=False)

    args = parser.parse_args()

    return args

