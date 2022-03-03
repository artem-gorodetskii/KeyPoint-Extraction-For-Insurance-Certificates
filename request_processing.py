#!/usr/bin/python3

import urllib3
import base64
import pickle
import json

from argparse import ArgumentParser

def make_argument_parser():
    """
    Function, where all possible program arguments are described
    :returns: ArgumentParser object
    """
    parser = ArgumentParser()

    parser.add_argument('-i', '--input-file',
                        dest='input_file',
                        help='path to the image of document',
                        metavar='FILE')

    parser.add_argument('-o', '--output-file',
                        dest='output_file',
                        help="JSON file with output information, if 'same' file will be saved to the same directory as input-file with .json extension",
                        metavar='FILE',
                        default='output_data.json')

    return parser


def is_input_correct(args):
    """
    Function checks correctness of input data.
    :param args: parsed arguments
    :return: boolean result, if all arguments are correct or not
    """

    is_correct = True

    if args.input_file is None and len(args) == 0:
        print('Error: Path to input file should be specified.')
        is_correct = False

    return is_correct

def main():

    # Creating parser object
    parser = make_argument_parser()
    # Parsing arguments
    args = parser.parse_args()
    # if input is not correct - exit  
    if not is_input_correct(args):  
        exit(-1)

    # Making request to server
    http = urllib3.PoolManager()
    r = http.request('POST', 'http://localhost:5000/get_entities',
                     fields={'image_path': base64.b64encode(pickle.dumps(args.input_file))})

    # Decode data
    data = pickle.loads(base64.b64decode(r.data))
    
    # Define path for json file
    if args.output_file != 'same':
        output_path = args.output_file
    else:
        output_path = args.input_file.split('.jpg')[0] + '.json'

    print(output_path)
    # Save json file
    with open(output_path, 'w') as f:
        f.write(data)

if __name__ == '__main__':
    main()
