"""
"""
from matplotlib.ft2font import LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH
from regex import E
from torch import _linalg_inv_out_helper_
from enc_pred.utils.spreadsheet import ExperimentSet, CalibrationExperiment, PredictionExperiment, \
    ExperimentTaskGroup, ExperimentDataSizeGroup, ExperimentModelGroup
from enc_pred.utils.table_utils import Row, Cell, MyTable
import argparse


CALIBRATION_METHOD = ['temperature-scaling', 'beta-calibration', 'gp-calibration', 'histogram-binning']
CM_DISPLAY_NAME = {
    'temperature-scaling': 'TS',
    'beta-calibration': 'Beta',
    'gp-calibration': 'GPcalib',
    'histogram-binning': 'HIST'
}

TASK_DISPLAY_NAME = {
    'pos_tags': 'POS',
    'deprel': 'UDP',
    'ner': 'NER',
    'xnli': 'XNLI'
}

LANG_LIST = ['en', 'de', 'fr', 'es', 'ru', 'hi', 'ar', 'zh']
TASK_LIST = ['pos_tags', 'deprel', 'ner', 'xnli']


def parse_args():
    """
    """
    parser = argparse.ArgumentParser(
        """Generating full result table w.r.t. a
        specific model setting.
        """
    )
    
    parser.add_argument(
        '--model_stem', action='store', dest='model_stem',
        type=str, required=True, help='Which setting to consider'
    )

    return parser.parse_args()


HEADER = r"""
    \begin{center}
    \begin{tabular}{ccr|rrrrrrrr}
"""

TAIL = r"""
    \end{tabular}
    \end{center}
"""

    
def main():
    args = parse_args()
    
    main_table = MyTable(
        header=HEADER,
        tail=TAIL
    )

    title_cell_list = [
        Cell(
            text='Task',
            rows=1,
            columns=1,
            style='bf'
        ),
        Cell(
            text='Metric(\\%)',
            rows=1,
            columns=1,
            style='bf'
        )
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
    
    for tidx, task in enumerate(TASK_LIST):
        prediction_experiments = PredictionExperiment.from_dir(
            dirname=f'runs/{args.model_stem}_{task}'
        )

        logit_list = ['logit']
        if task == 'deprel':
            logit_list = ['logit', 'selection_logit']
        
        for logit in logit_list:
            calibration_experiments = {cm: CalibrationExperiment.from_dir(f"runs/{args.model_stem}_{task}={cm}={logit}") for cm in CALIBRATION_METHOD}

            experiment_set = ExperimentSet(
                prediction_experiments,
                calibration_experiments=calibration_experiments
            )

            data = experiment_set.create_table()

            # try to construct table

            if logit == 'logit':
                main_table.insert_row(
                    Row(
                        cells=[
                            Cell(
                                text=TASK_DISPLAY_NAME[task],
                                style='it',
                                rows=1,
                                columns=1
                            ),
                            Cell(
                                text=list(data['ar']['prediction'].keys())[0],
                                style='it',
                                rows=1,
                                columns=1
                            )
                        ] + [
                            Cell(
                                text=list(data[lang]['prediction'].values())[0],
                                rows=1,
                                columns=1,
                            ) for lang in LANG_LIST
                        ],
                        before_row='\\midrule' if tidx == 0 else '\\cmidrule{1-10}',
                    )
                )
            
            prefix = ''

            if task == 'deprel':
                if logit == 'logit':
                    prefix = 'l-'
                else:
                    prefix = 'h-'
                    
            # first add a row of original scores
            main_table.insert_row(
                Row(
                    cells=[
                        Cell(
                            text=f'',
                            rows=1,
                            columns=1
                        ),
                        Cell(
                            text=f'{prefix}ECE',
                            rows=1,
                            columns=1
                        ),
                    ] + [
                        Cell(
                            text=data[lang]['calibration']['original']['mean'],
                            rows=1,
                            columns=1
                        ) for lang in LANG_LIST
                    ]
                )
            )
            # insert other calibration methods

            def _get_effect(s, d):
                if s:
                    if d == '+':
                        return '\\worse'
                    else:
                        return '\\better'
                else:
                    return None
            
            for cidx, cm in enumerate(CALIBRATION_METHOD):
                
                main_table.insert_row(
                    Row(
                        cells=[
                            Cell(
                                text=f'',
                                rows=1,
                                columns=1,
                            ),
                            Cell(
                                text=f"{prefix}{CM_DISPLAY_NAME[cm]}",
                                rows=1,
                                columns=1
                            ),
                        ] + [
                            Cell(
                                rows=1,
                                columns=1,
                                text=data[lang]['calibration'][cm]['mean'],
                                effect=_get_effect(data[lang]['calibration'][cm]['is_significant'], data[lang]['calibration'][cm]['direction'])
                            ) for lang in LANG_LIST
                        ]
                    )
                )

    main_table[-1].after_row = '\\bottomrule'
    print(main_table.render_rows())

        
if __name__ == '__main__':
    main()