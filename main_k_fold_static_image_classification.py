import argparse
from scripts.experiment_k_fold_static_image_classification import main_k_fold_static_image_classification

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-n_fold', type=int, help='How many folds in total?')
    parser.add_argument('-fold_to_run', nargs="+", type=int, help='Which fold(s) to run in this session?')
    parser.add_argument('-gpu', type=int, help='Which gpu to use?')
    parser.add_argument('-cpu_thread', type=int, help='How many threads are allowed?')
    args = parser.parse_args()

    main_k_fold_static_image_classification(args.n_fold, args.fold_to_run, args.gpu, args.cpu_thread)
