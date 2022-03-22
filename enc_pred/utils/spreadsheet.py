"""Implementing spreadsheets related configurations
"""
from typing import Text, Dict, List, Optional, Tuple, Any, Iterable
# CREDENTIAL PATH (DA_UPDATE_CEREDENTIALS)
CREDENTIAL_PATH = "/brtx/604-nvme2/zpjiang/encode_predict/data/.credentials/da_update_credentials.json"
LANGUAGE_STEP = 2
TYPE_TO_FEATURES = {
    'deprel': [
        {
            'suffix': '',
            'content': [
                {
                    'label': 'LAS',
                    'keys': [
                        'performance::LAS'
                    ]
                }
            ]
        },
        {
            'suffix': '-calibration-logit',
            'content': [
                {
                    'label': 'label-brier',
                    'keys': [
                        'ori::brier::score',
                        'scaled::brier::score'
                    ]
                },
                {
                    'label': 'label-ECE',
                    'keys': [
                        'ori::ece::ECE',
                        'scaled::ece::ECE'
                    ]
                }
            ]
        },
        {
            'suffix': '-calibration-selection_logit',
            'content': [
                {
                    'label': 'head-brier',
                    'keys': [
                        'ori::brier::score',
                        'scaled::brier::score'
                    ]
                },
                {
                    'label': 'head-ECE',
                    'keys': [
                        'ori::ece::ECE',
                        'scaled::ece::ECE'
                    ]
                }
            ]
        }
    ],
    'pos_tags': [
        {
            'suffix': '',
            'content': [
                {
                    'label': 'Acc',
                    'keys': [
                        'performance::accuracy'
                    ]
                }
            ]
        },
        {
            'suffix': '-calibration-logit',
            'content': [
                {
                    'label': 'brier',
                    'keys': [
                        'ori::brier::score',
                        'scaled::brier::score'
                    ]
                },
                {
                    'label': 'ECE',
                    'keys': [
                        'ori::ece::ECE',
                        'scaled::ece::ECE'
                    ]
                }
            ]
        },
    ],
    'xnli': [
        {
            'suffix': '',
            'content': [
                {
                    'label': 'Acc',
                    'keys': [
                        'performance::accuracy'
                    ]
                }
            ]
        },
        {
            'suffix': '-calibration-logit',
            'content': [
                {
                    'label': 'brier',
                    'keys': [
                        'ori::brier::score',
                        'scaled::brier::score'
                    ]
                },
                {
                    'label': 'ECE',
                    'keys': [
                        'ori::ece::ECE',
                        'scaled::ece::ECE'
                    ]
                }
            ]
        },
    ],
    'ner': [
        {
            'suffix': '',
            'content': [
                {
                    'label': 'F-1',
                    'keys': [
                        'performance::fscore'
                    ]
                }
            ]
        },
        {
            'suffix': '-calibration-logit',
            'content': [
                {
                    'label': 'brier',
                    'keys': [
                        'ori::brier::score',
                        'scaled::brier::score'
                    ]
                },
                {
                    'label': 'ECE',
                    'keys': [
                        'ori::ece::ECE',
                        'scaled::ece::ECE'
                    ]
                }
            ]
        },
    ]
}


class Sheet:
    def __init__(
        self, rows: int,
        columns: int
    ):
        """
        """
        self._rows = rows
        self._columns = columns

        self._sheet = [['' for _  in range(self._columns)] for _ in range(self._rows)]

    def resize(self, rows: int, columns: int):
        """A resizing will change the number of rows and columns,
        padding each row with '' or truncate.
        """
        if rows == -1:
            rows = self._rows
        if columns == -1:
            columns = self._columns

        if columns < self._columns:
            self._sheet = [r[:columns] for r in self._sheet]
            self._columns = columns
        elif columns > self._columns:
            self._sheet = [r + ['' for _ in range(columns - self._columns)] for r in self._sheet]

        if rows < self._rows:
            self._sheet = self._sheet[:rows]
            self._rows = rows
        elif rows > self._rows:
            self._sheet = self._sheet + [['' for _ in range(self._columns)] for _ in range(rows - self.rows)]

    def set_pos(self, ridx: int, cidx: int, val: Text):
        """This is the main local setting function.
        """
        self._sheet[ridx][cidx] = val

    def push_update(self, worksheet):
        """Push current update to worksheet.
        """
        worksheet.update(self._sheet)

    def __getitem__(self, x: int) -> List[int]:
        return self._sheet[x]