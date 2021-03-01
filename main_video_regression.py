import argparse
from scripts.experiment_k_fold_video_regression import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-n_fold', type=int, help='How many folds in total?', default=9)
    parser.add_argument('-fold_to_run', nargs="+", type=int, help='Which fold(s) to run in this session?', default=[0])
    parser.add_argument('-gpu', type=int, help='Which gpu to use?', default=0)
    parser.add_argument('-cpu_thread', type=int, help='How many threads are allowed?', default=1)
    args = parser.parse_args()

    main(args.n_fold, args.fold_to_run, args.gpu, args.cpu_thread)
