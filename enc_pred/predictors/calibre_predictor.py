"""Predictor for generating calibration prediction.
"""
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from typing import Tuple, Dict, List, Optional, Any, Union, Text


class SpanLabelPredictor(Predictor):
    """
    """
    def predict(
        self,
        spans: List[Tuple],
        labels: List[str],
        sentence: List[Text]
    ) -> JsonDict:
        """This is the standard prediction process.
        """
        pass