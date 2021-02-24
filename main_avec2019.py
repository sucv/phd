import argparse
from scripts.experiment_avec19_video_regression import generic_experiment_avec19_video_regression
from scripts.experiment_avec2019_video_regression_backup import main_avec2019

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-tc', help='Subjects\' country for training set: DE, HU, all', default="all")
    parser.add_argument('-vc', help='Subjects\' country for validation set: DE, HU, all', default="all")
    parser.add_argument('-m', help='Model: 2d1d, 2dlstm', default="2d1d")
    parser.add_argument('-e', help='a: arousal, v: valence, b: both', default="a")
    parser.add_argument('-lr', type=float, help='Which gpu to use?', default=1e-6)
    parser.add_argument('-p', type=int, help='Patience for learning rate changes', default=5)
    parser.add_argument('-gpu', type=int, help='Which gpu to use?', default=1)
    parser.add_argument('-cpu', type=int, help='How many threads are allowed?', default=1)
    parser.add_argument('-s', type=str, help='To indicate different experiment instances', default='0')
    args = parser.parse_args()

    experiment_handler = generic_experiment_avec19_video_regression(args)
    experiment_handler.experiment()

    # main_avec2019(args.tc, args.vc, args.gpu, args.cpu)
