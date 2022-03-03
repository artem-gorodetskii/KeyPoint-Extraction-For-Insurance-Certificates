#!/usr/bin/python3

from argparse import ArgumentParser
from flask import Flask, request
from entities_detector import EntitiesDetector
import pickle
import base64

app = Flask(__name__)

def make_argument_parser():
    """
    Function, where all possible program arguments are described
    :returns: ArgumentParser object
    """
    parser = ArgumentParser()

    parser.add_argument('-cp', '--checkpoint_path',
                        dest='checkpoint_path',
                        help='path to the checkpoint of pretrained model',
                        metavar='FILE',
                        default='pretrained_model/pretrained.pth')

    parser.add_argument('-d', '--device',
                        dest='device',
                        help="device for calculations, should be 'cpu' or 'cuda'",
                        default='cpu',
                        metavar='N')

    parser.add_argument('-tmp_dir', '--temporary_files_directory',
                        dest='temporary_files_directory',
                        help="name for directory for temporary files",
                        metavar='FILE',
                        default='server_temporary_data')

    return parser

@app.route('/get_entities', methods=['POST'])
def get_entities():
    """
    Server function.
    :return: encoded json file with predicted fields.
    """

    if request.method == 'POST':
        # read image path
        img_path = pickle.loads(base64.b64decode(request.form.get('image_path')))
        # predict entities with Pick model
        output = Detector.detect(img_path, output_path=None, unique_output_name=False, 
                                 save_output=False, return_output=True)
        # return encoded json file with predicted fields
        return base64.b64encode(pickle.dumps(output))


# Creating parser object
parser = make_argument_parser()
# Parsing arguments
args = parser.parse_args()

checkpoint_path = args.checkpoint_path
output_directory = args.temporary_files_directory

if args.device not in ['cpu', 'cuda']:
    device_type = 'cpu'
else:
    device_type = args.device

# Initializing EntitiesDetector instance
Detector = EntitiesDetector(checkpoint_path, device_type, output_directory)

# Starting server
app.run(host='0.0.0.0', port=5000) 
