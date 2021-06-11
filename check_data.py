from utils.data_loaders import show_samples
from config import parse_args

args = parse_args()
subset = 'train'

show_samples(args, subset)