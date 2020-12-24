import cvgutils.Viz as Viz
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deploying command')
    parser.add_argument('--fn',type=str, default='', help='figure pickle filename')
    args = parser.parse_args()

    fig = Viz.loadInteractiveFig('./renderout/fig.pickle')
    fig.show