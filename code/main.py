import argparse
from detection import train_detection
from categorization import train_categorization

# source: https://realpython.com/command-line-interfaces-python-argparse/

def main(args):
    printing_vars(args)

    if args.mode == "detect":
        train_detection(args)
    elif args.mode == "categorize":
        train_categorization(args)
    else:
        raise NotImplementedError("Not yet implemented")


def printing_vars(args):
    print(vars(args))
    print('Network:', args.model)
    print('Normalization:', args.norm)
    print('Height:', args.height)
    print('Batch:', args.batch_size)
    print('Dropout:', args.dropout)
    print("lr:", args.learning_rate)


def parse_args():
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', required=True, type=str, choices=["detect", "categorize"])
    parser.add_argument('--data-path', required=True, type=str, help='Main directory where images and CSVs are stored')
    parser.add_argument('--model', default='ResNet', type=str, choices=['ResNet', 'Mobile', 'VGG'], help='Convolutional Neural Network to train')
    parser.add_argument('--height', default=416, type=int, help='Input image height')
    parser.add_argument('--width', default=416, type=int, help='Input image width')
    parser.add_argument('--eye-only', action='store_true', help='Whether to consider only images with eyes. True if specified')
    parser.add_argument('--num-vars', action='store_true', help='Whether to use also numerical variables. True if specified')
    parser.add_argument('--norm', action='store_true', help='Whether to perform normalisation to the images. True if specified')
    parser.add_argument('--norm-mode', default='model', type=str, choices=['z-norm', 'model', 'simple', 'none'], help='Normalization mode to apply to images')
    parser.add_argument('--rot', action='store_true', help='Whether to rotate images. True if specified')
    parser.add_argument('--crop', action='store_true', help='Whether to perform random cropping to the images. True if specified')
    parser.add_argument('--crop-mode', default='uniform', type=str, choices=['uniform', 'weighted'], help='Random crop mode to apply to images')
    parser.add_argument('--nb-crops', default=1, type=int, help='Number of random crops to perform')
    parser.add_argument('--aug', action='store_true', help='Whether to perform data augmentation. True if specified')
    parser.add_argument('--batch-size', default=8, type=int, help='Batch size')
    parser.add_argument('--buffer-size', default=100, type=int, help='Buffer size')
    parser.add_argument('--epochs', default=30, type=int, help='Number of epochs to train')
    parser.add_argument('--learning-rate', default=1e-4, type=float, help='Learning rate to use in training')
    parser.add_argument('--splits', default=5, type=int, help='Number of splits to perform in StratifiedKFold')
    parser.add_argument('--dropout', action='store_true', help='Whether to use dropout in CNNs. True if specified')
    parser.add_argument('--drop-rate', default=0.5, type=float, help='Dropout rate to apply if dropout == True')
    parser.add_argument('--fine-tune', required=True, action='store_true', help='Whether to perform fine-tuning. True if specified')
    parser.add_argument('--fine-tune-at', default='last5', help='Options: int - Layer to start fine-tuning from; str - Number of last layers to fine-tune: "lastX"')
    parser.add_argument('--initial-epochs', default=20, type=int, help='Initial epochs')
    parser.add_argument('--ft-epochs', default=10, type=int, help='Epochs for fine-tuning stage')

    #args = vars(parser.parse_args())
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())