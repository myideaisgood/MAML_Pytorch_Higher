import argparse

def parse_training_args(parser):
    """Add args used for training only.

    Args:
        parser: An argparse object.
    """
    # Few Shot Parameters
    parser.add_argument('--N_way', type=int, default=5,
                        help='Number of classes')

    parser.add_argument('--K_shot', type=int, default=5,
                        help='Number of samples')

    parser.add_argument('--query_num', type=int, default=15,
                        help='Number of query samples per image')

    parser.add_argument('--evaluate_task', type=int, default=100,
                        help='Number of evaluation task')

    # MAML parameters
    parser.add_argument('--task_num', type=int, default=2,
                        help='Number of tasks in a batch of tasks')

    parser.add_argument('--num_steps_train', type=int, default=5,
                        help='Number of fast adaptation steps, ie. gradient descent in train')

    parser.add_argument('--num_steps_test', type=int, default=10,
                        help='Number of fast adaptation steps, ie. gradient descent in test')

    parser.add_argument('--step_size', type=float, default=0.01,
                        help='Step size of inner loop')

    parser.add_argument('--first-order', type=str2bool, default=True,
                        help='Use the first order approximation, do not use higher-order derivatives during meta-optimization')

    # Environment parameters
    parser.add_argument('--hidden_unit', type=int, default=64,
                        help='Hidden unit of model')

    # Session parameters
    parser.add_argument('--gpu_num', type=int, default=1,
                        help='GPU number to use')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Minibatch size')

    parser.add_argument('--epochs', type=int, default=60000,
                        help='Number of epochs to train')

    parser.add_argument('--meta_lr', type=float, default=1e-4,
                        help='Learning rate of meta learning')

    parser.add_argument('--summary_every', type=int, default=10,
                        help='How many iterations to do per TensorBoard summary write')

    parser.add_argument('--print_every', type=int, default=1,
                        help='How many iterations print for loss evaluation')

    parser.add_argument('--save_every', type=int, default=10,
                        help='How many iterations to save')                        

    parser.add_argument('--evaluate_every', type=int, default=10,
                        help='How many iterations to evaluate')

    # Directory parameters
    parser.add_argument('--data_dir', type=str, default="dataset/",
                        help='dataset directory')

    parser.add_argument('--dataset', type=str, default="miniImagenet",
                        help='miniImagenet')

    parser.add_argument('--experiment_name', type=str, default='gpu1/',
                        help='Experiment Name directory')

    parser.add_argument('--ckpt_dir', type=str, default="ckpt/",
                        help='checkpoint directory')

    parser.add_argument('--log_dir', type=str, default="log/",
                        help='log directory')

    parser.add_argument('--weights', type=str, default="ckpt-best.pth",
                        help='Fill In Fill In')

def parse_args():
    """Initializes a parser and reads the command line parameters.

    Raises:
        ValueError: If the parameters are incorrect.

    Returns:
        An object containing all the parameters.
    """

    parser = argparse.ArgumentParser(description='UNet')
    parse_training_args(parser)

    return parser.parse_args()

def str2bool(v):
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

if __name__ == '__main__':
    """Testing that the arguments in fact do get parsed
    """

    args = parse_args()
    args = args.__dict__
    print("Arguments:")

    for key, value in sorted(args.items()):
        print('\t%15s:\t%s' % (key, value))
