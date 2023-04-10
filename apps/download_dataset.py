import configargparse, urllib.request, os


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--ptb_pth',
                        type=str, default='./data/ptb',
                        help='path to store the Penn Treebank dataset')
    parser.add_argument('--ptb_data',
                        type=str, default='https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.',
                        help='link to download the Penn Treebank dataset')
    parser.add_argument("--cifar10_pth",
                        type=str, default='./data',
                        help='path to the CIFAR-10 dataset')
    parser.add_argument("--cifar10_data",
                        type=str, default='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                        help='path to the CIFAR-10 dataset')
    return parser.parse_args()

if __name__ == "__main__":
    args = config_parser()
    # Penn Treebank dataset
    if not os.path.exists(args.ptb_pth):
        os.system(f"mkdir -p {args.ptb_pth}")
    
    for f in ['train.txt', 'test.txt', 'valid.txt']:
        if not os.path.exists(os.path.join(args.ptb_pth, f)):
            urllib.request.urlretrieve(args.ptb_data + f, os.path.join(args.ptb_pth, f))
    # CIFAR-10 dataset
    if not os.path.isdir("./data/cifar-10-batches-py"):
        urllib.request.urlretrieve(args.cifar10_data, os.path.join(args.cifar10_pth, 'cifar-10-python.tar.gz'))
        os.system("tar -xvzf './data/cifar-10-python.tar.gz' -C './data'")
