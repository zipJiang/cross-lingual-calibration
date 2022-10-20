"""
"""
import os
import json
import numpy as np
from scipy.stats import ttest_ind
from enc_pred.utils.table_utils import Cell, Row, MyTable
from typing import *


parent_dir = '/brtx/604-nvme2/zpjiang/encode_predict/seqtag-runs/'
model_type = 'large-xlmr'

LANG_LIST = ['en', 'de', 'fr', 'es', 'ru', 'hi', 'ar', 'zh']
CM_LIST = ['temperature-scaling', 'gp-calibration']
DS_MAP = {
    '': 'full',
    '-lr': 'low-data',
    '-llr': 'very-low-data'
}
CM_MAP = {
    'temperature-scaling': "TS",
    'gp-calibration': 'GPcalib',
}
MODEL_LIST = ['-crf', '']

HEADER = r"""
    \begin{center}
    \begin{tabular}{cr|rrrrrrrr}
"""

TAIL = r"""
    \end{tabular}
    \end{center}
"""


TASK = 'wikiann'


def _read_directory(dir: Text) -> Tuple[Dict[Text, List[float]]]:
    """
    """
    result_dict_scaled = {l: [] for l in LANG_LIST}
    result_dict_ori =  {l: [] for l in LANG_LIST}
    
    for i in range(1, 11):
        for lang in LANG_LIST:
            path = os.path.join(dir, str(i), 'eval', f"{lang}.json")
            
            if os.path.isfile(path):
                with open(path ,'r', encoding='utf-8') as file_:
                    dict_ = json.load(file_)
                    result_dict_scaled[lang].append(dict_['scaled::ece::ECE'])
                    result_dict_ori[lang].append(dict_['ori::ece::ECE'])
                    
    return (result_dict_scaled, result_dict_ori)


def main():
    """
    """
    
    main_table = MyTable(
        header=HEADER,
        tail=TAIL
    )

    title_cell_list = [
        Cell(
            text='Source',
            rows=1,
            columns=1,
            style='bf'
        ),
    ] + [ Cell(
        text=f"\\multicolumn{{1}}{{c}}{{\\textbf{{{language}}}}}",
        rows=1,
        columns=1
    ) for language in LANG_LIST ]
    
    main_table.insert_row(
        Row(
            cells=title_cell_list,
            before_row='\\toprule'
        )
    )

    # a table with header
    prev_scaled = None
    for data_source in ['', '-lr', '-llr']:
        row = Row(
            cells=[
                Cell(
                    text=DS_MAP[data_source],
                    rows=1,
                    columns=1,
                    style='it'
                )
            ] + [Cell(
                    text=f'',
                    rows=1,
                    columns=1
                ) for _ in range(9)
            ],
            before_row='\\midrule',
            after_row='\\midrule'
        )
        main_table.insert_row(row)
        for crf in ['', '-crf']:
            for cidx, cm in enumerate(CM_LIST):
                scaled_dict, ori_dict = _read_directory(os.path.join(parent_dir, f"large-xlmr{data_source}-{TASK}{crf}={cm}=logit"))
                
                if cidx == 0:
                    # write original result
                    row = Row(
                        cells=[
                            Cell(
                                text=f'',
                                rows=1,
                                columns=1
                            ),
                            Cell(
                                text=CM_MAP[cm],
                                rows=1,
                                columns=1
                            )
                        ] + [
                            Cell(
                                text=sum(ori_dict[lang]) / len(ori_dict[lang]),
                                rows=1,
                                columns=1
                            ) for lang in LANG_LIST
                        ]
                    )
                    main_table.insert_row(row)
                
                # also need to write scaled_dict

                def _determine_effect(a: List[float], b: List[float]) -> Text:
                    """
                    """
                    direction = np.mean(a) < np.mean(b)
                    t_stats, pval = ttest_ind(
                        a=a,
                        b=b,
                        alternative='two-sided'
                    )

                    if pval < .05:
                        if direction:
                            return "\\better"
                        else:
                            return "\\worse"
                
                row = Row(
                    cells=[
                        Cell(
                            text=f'{crf}' if cidx == 0 else f'',
                            rows=1,
                            columns=1
                        ),
                        Cell(
                            text=CM_MAP[cm],
                            rows=1,
                            columns=1
                        )
                    ] + [
                        Cell(
                            text=sum(scaled_dict[lang]) / len(scaled_dict[lang]),
                            rows=1,
                            columns=1,
                            effect=_determine_effect(scaled_dict[lang], prev_scaled[lang]) if cidx != 0 else None
                        )
                        for lang in LANG_LIST
                    ]
                )
                main_table.insert_row(row)
                
                prev_scaled = scaled_dict
                
    main_table[-1].after_row = '\\bottomrule'

    print(main_table.render_rows())

                
if __name__ == '__main__':
    main()