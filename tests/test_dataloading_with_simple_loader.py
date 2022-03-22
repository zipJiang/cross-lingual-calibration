from unittest import TestCase
from enc_pred.data_readers.encode_predict_readers import WikiAnnReader, UniversalDependencyReader
from enc_pred.data_readers.sent_pred_reader import XNLIReader
from allennlp.data.data_loaders.simple_data_loader import SimpleDataLoader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer


class TestDataloadingWithSimpleLoader(TestCase):
    def setUp(self):
        """
        """
        self.test_path_wikiann = 'en/test'
        self.test_path_udparse = 'data/universal_dependency/ud-treebanks-v2.9/UD_English-Atis/en_atis-ud-dev.conllu'
        self.test_path_xnli = 'data/multinli_1.0/multinli_1.0_dev_matched.jsonl'

        self.batch_size = 4
        self.max_length = 32
        self.pretrained_model = 'bert-base-uncased'

        self.wikiann_reader = WikiAnnReader(
            max_length=self.max_length,
            pretrained_model=self.pretrained_model
        )
        self.udparse_reader = UniversalDependencyReader(
            max_length=self.max_length,
            pretrained_model=self.pretrained_model,
            task='pos_tags'
        )
        self.xnli_reader = XNLIReader(
            pretrained_model=self.pretrained_model,
            max_length=self.max_length
        )

    # def test_udparse_loading(self):
    #     """
    #     """
    #     vocabulary = Vocabulary.from_instances(
    #         self.udparse_reader.read(self.test_path_udparse)
    #     )

    #     dataloader = SimpleDataLoader(
    #         instances=list(self.udparse_reader.read(self.test_path_udparse)),
    #         batch_size=self.batch_size,
    #         vocab=vocabulary
    #     )

    #     for bidx, batch in enumerate(dataloader):

    #         batch = {key: val for key, val in batch.items() if key != 'raw_sentence'}
    #         print('-' * 20)
    #         print(batch)
    #         print('-' * 20)
            
    #         if bidx >= 5:
    #             break

    # def test_xnli_loading(self):
    #     """
    #     """
    #     vocabulary = Vocabulary.from_instances(
    #         self.xnli_reader.read(self.test_path_xnli)
    #     )

    #     dataloader = SimpleDataLoader(
    #         instances=list(self.xnli_reader.read(self.test_path_xnli)),
    #         batch_size=self.batch_size,
    #         vocab=vocabulary
    #     )

    #     tokenizer = PretrainedTransformerTokenizer(
    #         model_name='bert-base-uncased',
    #         add_special_tokens=False,
    #         max_length=32
    #     )

    #     for bidx, batch in enumerate(dataloader):
    #         print('-' * 20)
    #         if batch['tokens']['pieces']['mask'].shape != batch['tokens']['pieces']['token_ids'].shape:
    #             for idx, ins in enumerate(batch['meta-data']):
    #                 print(ins['premise'])
    #                 print(ins['hypothesis'])
    #                 print(batch['tokens']['pieces']['token_ids'][idx])
    #                 print(batch['tokens']['pieces']['mask'][idx])
    #         self.assertEqual(batch['tokens']['pieces']['mask'].shape, batch['tokens']['pieces']['token_ids'].shape)
    #         print('-' * 20)

    def test_single_xnli(self):
        """
        """
        vocabulary = Vocabulary.from_instances(
            self.xnli_reader.read(self.test_path_xnli)
        )
        premise = "yeah i i think my favorite restaurant is always been the one closest  you know the closest as long as it's it meets the minimum criteria you know of good food"
        hypothesis = "My favorite restaurants are always at least a hundred miles away from my house."
        # premise = 'hello my name is Zhengping'
        # hypothesis = 'hello, my name is Zhengping'
        instance = self.xnli_reader.text_to_instance(
            premise=premise,
            hypothesis=hypothesis,
            gold_label='contradiction'
        )

        print(len(instance['tokens']))
        print(instance['tokens'])

        instance['tokens'].index(vocabulary)
        print(instance['tokens'].get_padding_lengths())

    # def test_wikiann_loading(self):
    #     """
    #     """

    #     vocabulary = Vocabulary.from_instances(
    #         self.wikiann_reader.read(
    #             self.test_path_wikiann
    #         )
    #     )

    #     dataloader = SimpleDataLoader(
    #         instances=list(self.wikiann_reader.read(self.test_path_wikiann)),
    #         batch_size=self.batch_size,
    #         vocab=vocabulary
    #     )

    #     for bidx, batch in enumerate(dataloader):

    #         batch = {key: val for key, val in batch.items() if key != 'raw_sentence'}
    #         print('-' * 20)
    #         print(batch)
    #         print('-' * 20)
            
    #         if bidx >= 5:
    #             break