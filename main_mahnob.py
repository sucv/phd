import argparse
from scripts.experiment_mahnob_video_regression import  generic_experiment_mahnob_video_regression

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-n_fold', type=int, help='What job to run? 0: Regression Valence', default=5)
    parser.add_argument('-folds_to_run', nargs="+", type=int, help='Which fold(s) to run in this session?', default=[0])
    parser.add_argument('-m', help='Model: 2d1d, 2dlstm', default="2d1d")
    parser.add_argument('-j', type=int, help='What job to run? 0: Regression Valence', default=0)
    parser.add_argument('-modal',  nargs="*", default=['frame', 'eeg'])
    parser.add_argument('-lr', type=float, help='The initial learning rate.', default=1e-6)
    parser.add_argument('-d', type=float, help='Time delay between input and label, in seconds', default=0)
    parser.add_argument('-p', type=int, help='Patience for learning rate changes', default=5)
    parser.add_argument('-gpu', type=int, help='Which gpu to use?', default=0)
    parser.add_argument('-cpu', type=int, help='How many threads are allowed?', default=1)
    parser.add_argument('-s', type=str, help='To indicate different experiment instances', default='0')
    parser.add_argument('-model_load_path', type=str, help='The path to save the trained model ',
                        default='') #/scratch/users/ntu/su012/pretrained_model
    parser.add_argument('-model_save_path', type=str, help='The path to save the trained model ', default='') # /scratch/users/ntu/su012/trained_model
    args = parser.parse_args()

    experiment_handler = generic_experiment_mahnob_video_regression(args)
    experiment_handler.experiment()

