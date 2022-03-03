# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import pytesseract
from pytesseract import Output
import cv2
import re
import json

import torch
import model.pick as pick_arch_module

from data_utils import documents
from data_utils.pick_dataset import BatchCollateFn
from pathlib import Path

from utils.class_utils import keys_vocab_cls, iob_labels_vocab_cls

class EntitiesDetector:
    """
    This class includes instance of pretrained model and all required 
    preprocessing function for recognizing  Entities in document image.
    """
    def __init__(self, checkpoint_path: str, device_type: str, output_directory: str):
        """
        Initialize Detector.
        :checkpoint_path: str, path to pretrained model.
        :device_type: str, device for calculation 'cpu' or 'cuda'
        :output_directory: str, path to output json file and temporary data (.csv file)
        """

        self.output_directory = output_directory

        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)

        self.boxes_and_transcripts_path = os.path.join(self.output_directory, 'boxes_and_transcripts.csv')

        # load the model
        self.device = torch.device(device_type)    
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.config = self.checkpoint['config']
        self.state_dict = self.checkpoint['state_dict']

        self.pick_model = self.config.init_obj('model_arch', pick_arch_module)
        self.pick_model = self.pick_model.to(self.device)
        self.pick_model.load_state_dict(self.state_dict)
        self.pick_model = self.pick_model.eval()

        # callate function
        self.collate_func = BatchCollateFn(training=False)

        self.resized_image_size = tuple(self.config.config['train_dataset']['args']['resized_image_size'])

        self.entities_types = {1: 'Type',
                               2: 'Address',
                               3: 'Zip Code',
                               4: 'First Name',
                               5: 'Last Name',
                               6: 'Additionall Interest',
                               7: 'Policy Number',
                               8: 'Carrier',
                               9: 'Liability Limit',
                               10: 'Premium',
                               11: 'Effective Date',
                               12: 'Expiration Date',
                               13: 'Print Date'}

        self.ind2tag = {value: key for (key, value) in iob_labels_vocab_cls.stoi.items()}
        self.ind2char = {value: key for (key, value) in keys_vocab_cls.stoi.items()}

        self.input_keys = ['whole_image', 'relation_features', 'text_segments', 
                           'text_length', 'boxes_coordinate', 'mask', 'image_indexs']

    def remove_lines_in_img(self, img_path, length_threshold = 30, distance_threshold = 1.41421356, 
                            canny_th1 = 50.0, canny_th2 = 50.0, canny_aperture_size = 3):
        """
        Remove lines in document using opencv functions.
        :img_path: str, path to document image
        :length_threshold: float, Segment shorter than this will be discarded
        :distance_threshold: float, A point placed from a hypothesis line segment farther than this will be regarded as an outlier
        :canny_th1: float, First threshold for hysteresis procedure in Canny()
        :canny_th2: float, Second threshold for hysteresis procedure in Canny()
        :canny_aperture_size: float, Aperturesize for the sobel operator in Canny(). If zero, Canny() is not applied and the input image is taken as an edge image.
        """
    
        img = cv2.imread(img_path)

        # removing lines
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fld = cv2.ximgproc.createFastLineDetector(length_threshold, distance_threshold, canny_th1,
                                                  canny_th2, canny_aperture_size)
    
        lines = fld.detect(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
        if lines is not None:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(img,(x1,y1),(x2,y2),(255,255,255),4)
                
        return img

    def make_ocr_and_save(self, img_path):
        """
        Perform preprocessing steps and OCR by Tesseract.
        Function makes temporary .csv file with "self.boxes_and_transcripts_path" path.
        :img_path: str, path to document image
        """

        ocr_data = {'x1':[], 'y1':[], 'x2':[], 'y2':[], 'x3':[], 'y3':[], 'x4':[], 'y4':[], 'text':[]}

        img = self.remove_lines_in_img(img_path)
        ocr_output = pytesseract.image_to_data(img, output_type=Output.DICT)
        n_boxes = len(ocr_output['level'])

        for i in range(n_boxes):
            text = ocr_output['text'][i].strip()
            try:
                text = re.sub('_+','',text)
                text = re.sub(r'[^\w\s\/\-]', '', text.lower())
            except TypeError:
                continue
            if len(text) == 0:
                continue
            x, y, w, h = (ocr_output['left'][i], ocr_output['top'][i], ocr_output['width'][i], ocr_output['height'][i])
            ocr_data['x1'].append(x)
            ocr_data['y1'].append(y)
            ocr_data['x3'].append(x + w)
            ocr_data['y3'].append(y + h)
            ocr_data['text'].append(text)

        ocr_data['x2'] = ocr_data['x3']
        ocr_data['y2'] = ocr_data['y1']
        ocr_data['x4'] = ocr_data['x1']
        ocr_data['y4'] = ocr_data['y3']

        df = pd.DataFrame(ocr_data)
        pd.DataFrame(ocr_data).to_csv(self.boxes_and_transcripts_path, header=False)

    def detect(self, img_path, output_path=None, unique_output_name=True, 
              save_output=False, return_output=True):
        """
        Perform preprocessing steps and OCR by Tesseract.
        :img_path: str, path to document image
        :output_path: str, path for output .json file, actual if save_output==True.
        :unique_output_name: boolean, if True the output .json file will have the same name as input image, if False thr output 
        file will be output.json.
        :save_output: boolean, if True output will be saved as JSON file.
        :return_output: boolean, if True the function will return output in json format.
        """

        # perform OCR of document (image)
        self.make_ocr_and_save(img_path)

        boxes_and_transcripts_file = Path(self.boxes_and_transcripts_path)
        image_file = Path(img_path)

        if output_path is None:
            if unique_output_name:
                output_path = os.path.join(self.output_directory, image_file.stem + '.json')
            else:
                output_path = os.path.join(self.output_directory, 'output.json')

        # creating instance of Document class
        document = documents.Document(boxes_and_transcripts_file, image_file, 
                                      resized_image_size=self.resized_image_size,
                                      image_index=0, training=False)

        # define order of words
        segments_order = document.segments_order
        # preparing model input
        model_input = self.collate_func([document])

        for key in self.input_keys:
            if key in model_input:
                model_input[key] = model_input[key].to(self.device)

        # making mask indicated end of the word for each character
        text_length = model_input['text_length'][0].detach().cpu().numpy()
        end_of_word = []
        for length in text_length:
            end_of_word.extend([0]*(length-1))
            end_of_word.extend([1])

        # make predictions
        output = self.pick_model(**model_input)
        logits = output['logits']
        new_mask = output['new_mask']
        best_paths = self.pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
        # [0] because the batch size for test mode is 1
        predicted_tags, score = best_paths[0]

        B, N, T = model_input['text_segments'].size()
        doc_x = model_input['text_segments'].reshape(B, N * T)[0]
        doc_mask = model_input['mask'].reshape(B, N * T)[0]
        valid_doc_x = doc_x[doc_mask == 1].detach().cpu().numpy()

        # define detected entities
        detected_entities = {}
        for key in self.entities_types.values():
            detected_entities[key] = []

        # preparing output data for json
        # detected_entities with predicted characters
        counter = -1
        word = []

        for ind in predicted_tags:
            tag = self.ind2tag[ind]
            counter += 1
            if tag not in ['O', '<pad>', '<unk>']:
                tag = int(tag.split('-')[-1])
                entity_type = self.entities_types[tag]
                char = self.ind2char[valid_doc_x[counter]]
                word.append(char)
                if end_of_word[counter] == 1:
                    # write order of word in document and transcript
                    detected_entities[entity_type].append((segments_order[counter], ''.join(x for x in word)))
                    word = []

        for key in detected_entities.keys():
            # sort by order in document
            detected_entities[key] = sorted(detected_entities[key], key=lambda x: x[0])
            # combine words together 
            detected_entities[key] = ' '.join(x[-1] for x in detected_entities[key])
            detected_entities[key] = detected_entities[key].strip()

        if save_output:
            with open(self.output_path, "w") as outfile:
                json.dump(detected_entities, outfile, indent = 4)
                outfile.close()

        if return_output:
            return json.dumps(detected_entities, indent = 4)

       