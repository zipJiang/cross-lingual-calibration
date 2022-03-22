"""This script will check what's the parameter of
the model.
"""
from allennlp.models.model import Model
import enc_pred
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        """Check model parameter.
        """
    )
    parser.add_argument(
        '--archive_dir', action='store', dest='archive_dir',
        type=str, required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model = Model.from_archive(
        archive_file=args.archive_dir
    )

    for pname, param in model.named_parameters():
        print(f"{pname}: {1 / param.cpu().item()}")


if __name__ == '__main__':
    main()
