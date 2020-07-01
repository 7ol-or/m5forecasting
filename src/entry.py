import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument('experiment', type=str)
parser.add_argument('function', type=str)


if __name__ == "__main__":
    from config import FLAGS
    args = parser.parse_args()
    FLAGS.initialize(args.experiment)    
    FLAGS._define('FUNCTION', args.function)
    m = importlib.import_module(args.function)
    m.run()
