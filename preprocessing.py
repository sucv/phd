import argparse
import json
from utils.generic_preprocessing import *

if __name__ == '__main__':

    with open("configs/config_mahnob") as config_file:
        config = json.load(config_file)

    a = GenericPreprocessingMahnobFull(config)
