"""This script will be used to generate a subset of 
the English data for us train our model on.
"""
import argparse
import random
from collections import OrderedDict


def parse_args():
    """
    """
    parser = argparse.ArgumentParser(
        """Subsample a udparse document to give a subset
        of document according to categories.
        """
    )
    parser.add_argument(
        '--input_path', action='store', dest='input_path',
        type=str, required=True, help='Where to read the original document.'
    )
    parser.add_argument(
        '--write_to', action='store', dest='write_to',
        type=str, required=True, help='Where to write the subsampled data.'
    )
    parser.add_argument(
        '--num_samples', action='store', dest='num_samples',
        type=int, required=True, help='Where to write the subsampled dataset.'
    )
    parser.add_argument(
        '--seed', action='store', dest='seed',
        type=int, required=False, default=2265
    )

    return parser.parse_args()


def main():
    args = parse_args()

    doc_dict = OrderedDict()
    rand_obj = random.Random(args.seed)

    with open(args.input_path, 'r', encoding='utf-8') as file_:
        current_id = None
        current_entry = None
        for line in file_:
            # whether it is a comment
            line = line.strip()
            if line.startswith('#') and line[2:8] == 'newdoc':
                current_id = line[14:]
                current_type = current_id.split('-')[0]
                if current_type not in doc_dict:
                    doc_dict[current_type] = OrderedDict()
                doc_dict[current_type][current_id] = []
                current_entry = doc_dict[current_type][current_id]

            current_entry.append(line)

    # try to do classification
    type_counter = {type_: len(doc_dict[type_]) for type_ in doc_dict.keys()}

    total_number_document = sum([val for val in type_counter.values()])
    subsampled_type_counter = {key: int(val / total_number_document * args.num_samples) for key, val in type_counter.items()}
    total_subsampled_count = sum([val for val in subsampled_type_counter.values()])

    print(type_counter)

    difference = args.num_samples - total_subsampled_count
    per_type_compensation = difference // len(subsampled_type_counter)
    singled_out_compensation = difference % len(subsampled_type_counter)

    for key in subsampled_type_counter.keys():
        subsampled_type_counter[key] += per_type_compensation

    subsampled_type_counter[rand_obj.choice(
        [key for key in subsampled_type_counter.keys()
         if type_counter[key] - subsampled_type_counter[key] >= singled_out_compensation]
    )] += singled_out_compensation

    print(subsampled_type_counter)

    with open(args.write_to, 'w', encoding='utf-8') as file_:
        for key in doc_dict:
            num_type_sample = subsampled_type_counter[key]
            selected_items = rand_obj.sample(list(doc_dict[key].values()), num_type_sample)

            # write items back
            for writing_item in selected_items:
                for writing_line in writing_item:
                    file_.write(writing_line + '\n')


if __name__ == '__main__':
    """Running the dataset sampling script.
    """
    main()