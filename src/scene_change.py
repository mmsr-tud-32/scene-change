import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('foreground')
    parser.add_argument('background')

    return parser.parse_args()


def scene_change(foreground, background):
    
    pass


if __name__ == "__main__":
    config = vars(get_arguments())
    scene_change(**config)
