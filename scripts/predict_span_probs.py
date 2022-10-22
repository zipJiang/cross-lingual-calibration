"""Predict span probability from a CRF model
"""
import torch
import json
from enc_pred.data_readers.smooth_sequence_tagging_loaders import SmoothNERTaggingReader
from enc_pred.data_readers.sequence_tagging_loaders import NERTaggingReader, POSTaggingReader
from allennlp.models.model import Model
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.data.vocabulary import Vocabulary
from typing import Text, List, Tuple, Dict, Any, Union
from tqdm import tqdm
from enc_pred.models.plug_and_play_crf import PlugAndPlayCRF
import os
import argparse
import random
import numpy as np


random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.manual_seed(42)


class SpanProbPredictor:

    def __init__(
        self,
        model: PlugAndPlayCRF,
        vocab: Vocabulary,
        dataset_reader: DatasetReader,
        data_path: Union[Text, List[Text]],
        batch_size: int = 16,
        device: int = 0,
    ):
        self.model = model
        self.dataset_reader = dataset_reader
        self.data_path = data_path
        self.device = device
        self.batch_size = batch_size
        self.vocab = vocab
        
        self.model.to(self.device)
        self.model.eval()

    def predict(self) -> List[Dict[Text, Any]]:
        
        def detach_tensor_dict(tdict: Dict[Text, torch.Tensor]) -> Dict[Text, List[float]]:
            """
            """
            return {
                'logits': tdict['logits'].cpu().detach().tolist(),
                'labels': tdict['labels'].cpu().detach().tolist()
            }

        dataloader = MultiProcessDataLoader(
            reader=self.dataset_reader,
            data_path=self.data_path,
            batch_size=self.batch_size,
            shuffle=False,
            cuda_device=self.device
        )
        dataloader.index_with(self.vocab)
        
        result_list = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                output = self.model.calculate_span_probabilities(
                    **batch
                )
                output = detach_tensor_dict(output)
                
                for logit, label in zip(output['logits'], output['labels']):
                    result_list.append({
                        'logit': logit,
                        'label': label
                    })

        return result_list
                    
                    
def parse_args():
    parser = argparse.ArgumentParser(
        """Predict span logits by using the span probability extraction
        mechanism.
        """
    )
    
    parser.add_argument(
        "--archive_path", action='store', dest='archive_path',
        type=str, required=True, help='Where to load the model.'
    )

    parser.add_argument(
        '--data_config_path', action='store', dest='data_config_path',
        type=str, required=True, help='Where to load the dataset.'
    )
    
    parser.add_argument(
        '--device', action='store', dest='device',
        type=int, required=False, default=0, help='Where to put the procesing unit.'
    )
    
    parser.add_argument(
        '--output_path', type=str,
        required=True, help='Where to write the generation.'
    )
    
    parser.add_argument(
        '--task', type=str,
        required=True, help="Which task are we predicting?"
    )
    
    return parser.parse_args()


def main():

    args =  parse_args()
    
    archive_path: Text = args.archive_path
    if archive_path.endswith('/'):
        archive_path = archive_path[:-1]
        
    archive_name = os.path.basename(archive_path)
    # if 'large' in archive_name:
    #     pretrained_model = 'xlm-roberta-large'
    # elif 'xlmr' in archive_name:
    #     pretrained_model = 'xlm-roberta-base'
    # else:
    #     pretrained_model = 'bert-base-multilingual-cased'
    with open(os.path.join(archive_path, "config.json"), 'r', encoding='utf-8') as file_:
        pretrained_model = json.load(file_)['dataset_reader']['pretrained_model']


    if args.task == 'udparse' :
        dataset_reader = POSTaggingReader(
            max_length=512,
            pretrained_model=pretrained_model,
            is_prediction=False,
        )
        
    elif args.task == 'wikiann':
        dataset_reader = SmoothNERTaggingReader(
            pretrained_model=pretrained_model,
            max_length=32,  # use a small max_length
            max_span_length=8
        )
        
    model = Model.from_archive(
        args.archive_path
    )
    
    vocab = model.vocab

    with open(args.data_config_path, 'r', encoding='utf-8') as file_:
        dconfig = json.load(file_)
        print(dconfig)
        
    predictor = SpanProbPredictor(
        model=model,
        vocab=vocab,
        dataset_reader=dataset_reader,
        data_path=dconfig['file_path'][0],
        device=args.device
    )

    output = predictor.predict()
    
    with open(args.output_path, 'w', encoding='utf-8') as file_:
        for item in output:
            file_.write(json.dumps(item) + '\n')
                
                
if __name__ == '__main__':
    main()