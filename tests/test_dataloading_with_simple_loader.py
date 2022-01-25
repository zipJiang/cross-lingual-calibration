from unittest import TestCase
from enc_pred.data_readers.encode_predict_readers import WikiAnnReader, UniversalDependencyReader
from allennlp.data.data_loaders.simple_data_loader import SimpleDataLoader
from allennlp.data.vocabulary import Vocabulary


class TestDataloadingWithSimpleLoader(TestCase):
    def setUp(self):
        """
        """
        self.test_path_wikiann = 'en/test'
        self.test_path_udparse = 'data/universal_dependency/ud-treebanks-v2.9/UD_English-Atis/en_atis-ud-dev.conllu'

        self.batch_size = 4
        self.max_length = 256
        self.pretrained_model = 'bert-base-uncased'

        self.wikiann_reader = WikiAnnReader(
            max_length=self.max_length,
            pretrained_model=self.pretrained_model
        )
        self.udparse_reader = UniversalDependencyReader(
            max_length=self.max_length,
            pretrained_model=self.pretrained_model
        )

    def test_udparse_loading(self):
        """
        """
        vocabulary = Vocabulary.from_instances(
            self.udparse_reader.read(self.test_path_udparse)
        )

        dataloader = SimpleDataLoader(
            instances=list(self.udparse_reader.read(self.test_path_udparse)),
            batch_size=self.batch_size,
            vocab=vocabulary
        )

        for bidx, batch in enumerate(dataloader):

            batch = {key: val for key, val in batch.items() if key != 'raw_sentence'}
            print('-' * 20)
            print(batch)
            print('-' * 20)
            
            if bidx >= 5:
                break

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