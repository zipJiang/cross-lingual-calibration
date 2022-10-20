"""Bootstrap from dataset,
used to generate training dataset
for items.
"""
import json
import random
import argparse
import math


def parse_args():
    parser = argparse.ArgumentParser(
        """Bootstrap from dataset for the training dataset.
        """
    )
    parser.add_argument(
        '--src', action='store', dest='src',
        type=str, required=True,
        help='Where to load the source dataset.'
    )
    
    parser.add_argument(
        '--tgt', action='store', dest='tgt',
        type=str, required=True,
        help='Where to load the target dataset.'
    )
    
    parser.add_argument(
        '--ratio', action='store', dest='ratio',
        type=float, default=1
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    with open(args.src, 'r', encoding='utf-8') as file_:
        items = [line for line in file_]
        num_samples = math.floor(len(items) * args.ratio)
        
    with open(args.tgt, 'w', encoding='utf-8') as file_:
        items = random.choices(items, k=num_samples)
        for line in items:
            file_.write(line)
            
            
if __name__ == '__main__':
    main()