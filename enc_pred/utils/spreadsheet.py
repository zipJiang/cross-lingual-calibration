"""Implementing spreadsheets related configurations
"""
import os
import json
import numpy as np
import abc
import copy
from scipy.stats import ttest_ind
from typing import Text, Dict, List, Optional, Tuple, Any, Iterable, Union

from torch import maximum
from .commons import T_TEST_STATISTICS

# CREDENTIAL PATH (DA_UPDATE_CEREDENTIALS)
CREDENTIAL_PATH = "/brtx/604-nvme2/zpjiang/encode_predict/data/.credentials/da_update_credentials.json"


RESULT_SCHEMA = {
    'deprel': {
        'prediction': {
            'key': 'LAS',
            'original_key': 'performance::LAS',
        },
        'calibration': [
            {
                'source': 'logit',
                'key': 'scaled',
                'original_key': 'scaled::ece::ECE'
            },
            {
                'source': 'logit',
                'key': 'original',
                'original_key': 'ori::ece::ECE'
            },
            {
                'source': 'selection_logit',
                'key': 'scaled',
                'original_key': 'scaled::ece::ECE'
            },
            {
                'source': 'selection_logit',
                'key': 'original',
                'original_key': 'ori::ece::ECE'
            }
        ]
    },
    'xnli': {
        'prediction': {
            'key': 'Acc',
            'original_key': 'performance::accuracy'
        },
        'calibration': [
            {
                'source': 'logit',
                'key': 'scaled',
                'original_key': 'scaled::ece::ECE'
            },
            {
                'source': 'logit',
                'key': 'original',
                'original_key': 'ori::ece::ECE'
            }
        ]
    },
    'ner': {
        'prediction': {
            'key': 'F-1',
            'original_key': 'performance::fscore'
        },
        'calibration': [
            {
                'source': 'logit',
                'key': 'scaled',
                'original_key': 'scaled::ece::ECE'
            },
            {
                'source': 'logit',
                'key': 'original',
                'original_key': 'ori::ece::ECE'
            }
        ]
    },
    'pos_tags': {
        'prediction': {
            'key': 'Acc',
            'original_key': 'performance::accuracy'
        },
        'calibration': [
            {
                'source': 'logit',
                'key': 'scaled',
                'original_key': 'scaled::ece::ECE'
            },
            {
                'source': 'logit',
                'key': 'original',
                'original_key': 'ori::ece::ECE'
            }
        ]
    }
}


def calculate_dist(eval_dict_list: List[Dict[Text, float]]) -> Dict[Text, Union[Dict[Text, Union[float, int, List[float]]], bool]]:
    """Giving a list of items that contains named
    field of floats, convert them to empirical distributions
    with field distribution (mean, var).
    """

    metric_names = eval_dict_list[0].keys()
    
    result = {}

    for metric in metric_names:
        val_list = np.array([eval_dict[metric] for eval_dict in eval_dict_list], dtype=np.float32)
        
        mean = val_list.mean().item()
        median = np.median(val_list).item()
        max_ = np.max(val_list).item()
        min_ = np.min(val_list).item()
        std = val_list.std(ddof=1).item()
        quantiles = np.quantile(val_list, q=[.25, .75]).tolist()
        num_samples = val_list.size
        
        result[metric] = {
            'mean': mean,
            'std': std,
            'min': min_,
            'max': max_,
            'median': median,
            'quartiles': quantiles,
            'num_samples': num_samples,
            'samples': val_list.tolist()
        }

    t_stats, pval = ttest_ind(
        a=result['scaled::ece::ECE']['samples'],
        b=result['ori::ece::ECE']['samples'],
        alternative='two-sided'
    )
    
    result['is_significant'] = pval < .05
    result['direction'] = np.sign(result['scaled::ece::ECE']['mean'] - result['ori::ece::ECE']['mean']).item()
        
    return result


class Experiment(abc.ABC):
    def __init__(self):
        """
        """
        pass
    
    @abc.abstractmethod
    def parse_result(self) -> Dict[Text, Any]:
        """
        """
        raise NotImplementedError
    
    @abc.abstractclassmethod
    def from_dir(cls, dirname: Text) -> "Experiment":
        """
        """
        raise NotImplementedError

    
class PredictionExperiment(Experiment):
    """
    """
    def __init__(
        self,
        parent: Text,
        model: Text,
        data_size: Text,
        task: Text,
    ):
        """
        """
        super().__init__()
        self._parent = parent
        self._model = model
        self._data_size = data_size
        self._task = task
        
        self.result = self.parse_result()

    def parse_result(self) -> Dict[Text, Any]:
        """
        """
        eval_dir = os.path.join(self.path, 'eval')
        assert os.path.isdir(eval_dir), f"{eval_dir} is not a directory!"
        
        result = {}

        for filename in os.listdir(eval_dir):
            lang = filename.split('.')[0]
            filepath = os.path.join(eval_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file_:
                eval_dict: Dict[Text, Any] = json.load(file_)

                lang_dict = {key: val for key, val in eval_dict.items() if key.startswith('performance')}
                result[lang] = lang_dict
                
        return result
    
    @classmethod
    def from_dir(cls, dirname: Text) -> "PredictionExperiment":
        """
        """
        base, dirname = os.path.split(dirname)
        splitted_dirname = dirname.split('_')
        model = splitted_dirname[0]
        task = dirname[len(model) + 1:]
        
        if model.endswith('-lr'):
            data_size = '-lr'
            model = model[:-3]
        elif model.endswith('-llr'):
            data_size = '-llr'
            model = model[:-4]
        else:
            data_size = ''
            
        return cls(
            parent=base,
            model=model,
            data_size=data_size,
            task=task
        )
        
    @property
    def basename(self) -> Text:
        """
        """
        return f"{self._model}{self._data_size}_{self._task}"

    @basename.setter
    def basename(self, x):
        """
        """
        raise NotImplementedError
    
    @property
    def path(self) -> Text:
        """
        """
        return os.path.join(self._parent, self.basename)

    def create_entry(self) -> Dict[Text, Union[Text, Dict]]:
        """
        """
        task_schema = RESULT_SCHEMA[self._task]
        result = {}
        element_schema = task_schema['prediction']

        for lang, lang_result in self.result.items():
            result[lang] = {}
            result[lang][element_schema['key']] = lang_result[element_schema['original_key']]
            
        # print(result)
        return result
    

class CalibrationExperiment:
    """
    """
    def __init__(
        self,
        parent: Text,
        model: Text,
        data_size: int,
        task: Text,
        method: Text,
        source: Text,
        maximum_experiment_num: int = 10
    ):
        super().__init__()
        self._parent = parent
        self._model = model
        self._data_size = data_size
        self._task = task
        self._method = method
        self._source = source
        self._maximum_experiment_num = maximum_experiment_num  # inclusive
        
        self.result = self.parse_result()
        
    def parse_result(self) -> Dict[Text, Any]:
        """Generate the directory result by reading
        the 0-10 experiments directory.
        """
        
        eval_result_list = []

        for i in range(1, self._maximum_experiment_num + 1):
            # if not os.path.isfile(os.path.join(self.path, str(i), 'best.th')):
            #     continue

            eval_dir = os.path.join(self.path, str(i), 'eval')
            assert os.path.isdir(eval_dir), f"{eval_dir} is not a directory!"

            eval_result = self._parse_eval_dir(eval_dir)
            eval_result_list.append(eval_result)
        
        # calculate result_distribution
        assert len(eval_result_list) > 0, 'f{len(eval_result_list)} shows that list is empty.'
        langs = eval_result_list[0].keys()
        
        print(self.path)

        lang_combined = {l: calculate_dist([eval_result_dict[l] for eval_result_dict in eval_result_list]) for l in langs}
        
        return lang_combined
            
    def _parse_eval_dir(self, eval_dir: Text) -> Dict[Text, Any]:
        """Return a evaluation directory of all experiments from the eval directory
        """
        result_dict = {}
        for filename in os.listdir(eval_dir):
            lang = filename.split('.')[0]
            lang_result_dict = self._parse_lang_result(os.path.join(eval_dir, filename))

            result_dict[lang] = lang_result_dict
            
        return result_dict
            
    def _parse_lang_result(self, filepath: Text) -> Dict[Text, Any]:
        """
        """
        with open(filepath, 'r', encoding='utf-8') as file_:
            result_dict = json.load(file_)

        return {
            'ori::ece::ECE': result_dict["ori::ece::ECE"],
            'scaled::ece::ECE': result_dict["scaled::ece::ECE"]
        }
    
    @property
    def basename(self) -> Text:
        """basename without property
        """
        return f"{self._model}{self._data_size}_{self._task}={self._method}={self._source}"

    @basename.setter
    def basename(self, x) -> Text:
        """
        """
        raise NotImplementedError

    @property
    def path(self) -> Text:
        """
        """
        return os.path.join(self._parent, self.basename)

    @path.setter
    def path(self, x) -> Text:
        """
        """
        raise NotImplementedError

    @classmethod
    def from_dir(cls, dirname: Text) -> "CalibrationExperiment":
        while dirname.endswith('/'):
            dirname = dirname[:-1]

        base, dirname = os.path.split(dirname)
        sections = dirname.split('=')
        splitted = sections[0].split('_')
        model = splitted[0]
        task = sections[0][len(splitted[0]) + 1:]
        
        if model.endswith('-lr'):
            data_size = '-lr'
            model = model[:-3]
        elif model.endswith('-llr'):
            data_size = '-llr'
            model = model[:-4]
        else:
            data_size = ''
        
        method = sections[1]
        source = sections[2]
        
        return cls(
            parent=base,
            model=model,
            data_size=data_size,
            task=task,
            method=method,
            source=source
        )
        
    def create_entry(self) -> Dict[Text, Union[Text, Dict]]:
        """
        """
        task_schema = RESULT_SCHEMA[self._task]
        result = {}
        for lang, lang_result in self.result.items():
            result[lang] = {}
            for element_dict in task_schema['calibration']:
                if element_dict['source'] != self._source:
                    continue
                extraction = lang_result[element_dict['original_key']]
                
                result[lang][element_dict['key']] = {
                    'mean': extraction['mean'],
                    'std': extraction['std'],
                    'max': extraction['max'],
                    'min': extraction['min'],
                    'quartiles': extraction['quartiles'],
                    'samples': extraction['samples']
                }
            result[lang].update({
                'direction': lang_result['direction'],
                'is_significant': lang_result['is_significant']
            })
            
        return result


# class Sheet:
#     def __init__(
#         self, rows: int,
#         columns: int
#     ):
#         """
#         """
#         self._rows = rows
#         self._columns = columns

#         self._sheet = [['' for _  in range(self._columns)] for _ in range(self._rows)]

#     def resize(self, rows: int, columns: int):
#         """A resizing will change the number of rows and columns,
#         padding each row with '' or truncate.
#         """
#         if rows == -1:
#             rows = self._rows
#         if columns == -1:
#             columns = self._columns

#         if columns < self._columns:
#             self._sheet = [r[:columns] for r in self._sheet]
#             self._columns = columns
#         elif columns > self._columns:
#             self._sheet = [r + ['' for _ in range(columns - self._columns)] for r in self._sheet]

#         if rows < self._rows:
#             self._sheet = self._sheet[:rows]
#             self._rows = rows
#         elif rows > self._rows:
#             self._sheet = self._sheet + [['' for _ in range(self._columns)] for _ in range(rows - self.rows)]

#     def set_pos(self, ridx: int, cidx: int, val: Text):
#         """This is the main local setting function.
#         """
#         self._sheet[ridx][cidx] = val

#     def push_update(self, worksheet):
#         """Push current update to worksheet.
#         """
#         worksheet.update(self._sheet)

#     def __getitem__(self, x: int) -> List[int]:
#         return self._sheet[x]




class ExperimentSet:
    """Experiment set that combines original
    experiments and calibration experiment set.
    """
    def __init__(
        self,
        prediction_experiment: PredictionExperiment,
        calibration_experiments: Dict[Text, CalibrationExperiment]
    ):
        self.prediction_experiment = prediction_experiment
        self.calibration_experiments = calibration_experiments

        # assert that the are of the same model and task and data_size
        model_list = [e._model for e in self.calibration_experiments.values()]
        task_list = [e._task for e in self.calibration_experiments.values()]
        data_size_list = [e._data_size for e in self.calibration_experiments.values()]

        assert all([val == self.prediction_experiment._model for val in model_list]), f'Calibration models should all be from {self.prediction_experiment._model}!'
        assert all([val == self.prediction_experiment._task for val in task_list]), f'Calibration task should all be from {self.prediction_experiment._task}!'
        assert all([val == self.prediction_experiment._data_size for val in data_size_list]), f'Calibration data_size should all be from {self.prediction_experiment._data_size}!'
        
    def create_table(self) -> Dict:
        """
        """
        calibration_result = {}
        for key, experiment in self.calibration_experiments.items():
            calibration_result[key] = \
                experiment.create_entry()
            
        predictions = self.prediction_experiment.create_entry()
        calibrations = calibration_result
        
        # poping the langauge to the first depth
        reformatting = {}
        for lang in predictions.keys():
            
            for key, val in calibrations.items():
                if 'zh' not in val:
                    print(key)

            calibration_dict = {key: val[lang] for key, val in calibrations.items()}
            original_dict = None
            
            for key in calibration_dict:
                if original_dict is None:
                    original_dict = copy.deepcopy(calibration_dict[key]['original'])
                else:
                    # print(key, original_dict['mean'] - calibration_dict[key]['original']['mean'])
                    assert abs(original_dict['mean'] - calibration_dict[key]['original']['mean']) < 1e-4, "original result from different runs not equal!"
                calibration_dict[key]['scaled']['is_significant'] = calibration_dict[key]['is_significant']
                calibration_dict[key]['scaled']['direction'] = calibration_dict[key]['direction']
                calibration_dict[key] = calibration_dict[key]['scaled']
            
            # original_dict['direction'] = 0.
            # original_dict['is_significant'] = False
            calibration_dict['original'] = original_dict

            reformatting[lang] = {
                'prediction': predictions[lang],
                'calibration': calibration_dict
            }

        return reformatting

        
class Group(abc.ABC):
    """The abstract grouping criteria.
    """
    def __init__(
        self,
        sub_groups: Dict[Text, Union["Group", ExperimentSet]],
    ):
        """Should contain a metadata field corresponding
        to 'sub_group_type': {data_size, task, model} etc.
        """
        self._sub_groups = sub_groups
        
    def create_table(self) -> Dict:
        """Create a list that can be easily rendered as tables
        """

        data = {key: group.create_table() for key, group in self._sub_groups.items()}
        return data
    
        
class ExperimentTaskGroup(Group):
    """Experiment group containing all of the
    experiments for a specific model specification.
    """
    def __init__(
        self,
        sub_groups: Dict[Text, "Group"]
    ):  
        super().__init__(sub_groups)
        
class ExperimentDataSizeGroup(Group):
    """Experiment group containing all of the experiments for a
    specific data_size.
    """
    def __init__(
        self,
        sub_groups: Dict[Text, "Group"]
    ):  
        super().__init__(sub_groups)
        
class ExperimentModelGroup(Group):
    """Contains result for a specific model that 
    can be constructed for a set of different
    datasize groups.
    """
    def __init__(
        self,
        sub_groups: Dict[Text, "Group"]
    ):  
        super().__init__(sub_groups)