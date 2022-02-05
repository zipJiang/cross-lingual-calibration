"""This script works with the Google Spreadsheet api to
push the experiments result to the sheet.
"""
import json
from enc_pred.utils.spreadsheet import CREDENTIAL_PATH, LANGUAGE_STEP, TYPE_TO_FEATURES, Sheet
import argparse
from typing import Tuple, Text, Union, List, Dict, Any, Optional

import gspread
import os


def parse_args():
    """
    """
    parser = argparse.ArgumentParser(
        """Pushing experiments result to google.
        """
    )

    ###

    parser.add_argument(
        "--runs", action='store', dest='runs',
        type=str, nargs='+',
        required=True
    )
    parser.add_argument(
        "--tasks", action='store', dest='tasks',
        type=str, required=True,
        choices=TYPE_TO_FEATURES.keys(),
        nargs='+'
    )
    parser.add_argument(
        '--sheetname', type=str, required=True,
        action='store', dest='sheetname'
        )
    parser.add_argument(
        '--precision', type=int, required=False,
        action='store', dest='precision',
        default=5
    )
    ###

    return parser.parse_args()


# TODO: add typing annotation to the function
def merge_cells(
    row_spans: List[Tuple[int, int]],
    col_spans: List[Tuple[int, int]],
    spreadsheet,
    worksheet, 
):
    """
    """
    body = {
        'requests': [
            {
                'mergeCells': {
                    'mergeType': 'MERGE_ALL',
                    'range': {
                        'sheetId': worksheet._properties['sheetId'],
                        'startRowIndex': rspan[0],
                        'endRowIndex': rspan[1],
                        'startColumnIndex': cspan[0],
                        'endColumnIndex': cspan[1]
                    }
                }
            } for rspan, cspan in zip(row_spans, col_spans)
        ]
    }

    res = spreadsheet.batch_update(
        body
    )

    return res


def main():
    """
    """
    args = parse_args()

    gc = gspread.service_account(filename=CREDENTIAL_PATH)
    sht1 = gc.open("Domain Adaptation Calibration Results")

    work_sheet = sht1.add_worksheet(title=args.sheetname, rows=100, cols=50)

    # now start writing result to the sheet
    row_indicator = 1
    language_dict = {}
    overall_metric_dict = {}
    furthest_right = 2
    merge_rspans = []
    merge_cspans = []

    # we first store a local list of values and then do batch_update to the cell.
    local_sheet: Sheet = Sheet(rows=100, columns=50)

    for rdir, task in zip(args.runs, args.tasks):
        local_metric_dict = {}
        local_sheet.set_pos(ridx=row_indicator, cidx=0, val=rdir)
        used_row = 0
        for specification in TYPE_TO_FEATURES[task]:
            suffix = specification['suffix']
            content = specification['content']

            for filename in os.listdir(os.path.join(f'{rdir}{suffix}', 'eval')):
                with open(os.path.join(f'{rdir}{suffix}', 'eval', filename), 'r', encoding='utf-8') as file_:
                    lang = filename.split('.')[0]
                    eval_dict = json.load(file_)

                    if lang not in language_dict:
                        language_dict[lang] = furthest_right
                        local_sheet.set_pos(ridx=0, cidx=language_dict[lang], val=lang)
                        furthest_right += LANGUAGE_STEP

                    # write data to file
                    for item in content:
                        # write label
                        if item['label'] not in local_metric_dict:
                            local_metric_dict[item['label']] = row_indicator + used_row
                            local_sheet.set_pos(cidx=1, ridx=local_metric_dict[item['label']], val=item['label'])
                            used_row += 1

                        for kidx, key in enumerate(item['keys']):
                            local_sheet.set_pos(
                                ridx=local_metric_dict[item['label']],
                                cidx=language_dict[lang] + kidx,
                                val=f"{eval_dict[key]:.{args.precision}f}")

        overall_metric_dict[rdir] = local_metric_dict
        merge_rspans.append((row_indicator, row_indicator + used_row))
        merge_cspans.append((0, 1))
        row_indicator += used_row

    # push the sheet changes to remote
    local_sheet.push_update(worksheet=work_sheet)
    # do formatting
    work_sheet.format(
        "1:1", {
            'backgroundColor': {
                "red": 1.,
                "green": 1.,
                "blue": 1.
            },
            'horizontalAlignment': "CENTER",
            'textFormat': {
                'bold': True
            }
        }
    )

    work_sheet.format(
        "A:A", {
            'backgroundColor': {
                "red": 1.,
                "green": 1.,
                "blue": 1.
            },
            'horizontalAlignment': "CENTER",
            'textFormat': {
                'bold': True
            }
        }
    )

    # generate merge specs
    for lval in language_dict.values():
        merge_rspans.append((0, 1))
        merge_cspans.append((lval, lval + 2))

        for local_dict in overall_metric_dict.values():
            for performance_val in local_dict.values():
                if local_sheet[performance_val][lval + 1] == '':
                    merge_rspans.append((performance_val, performance_val + 1))
                    merge_cspans.append((lval, lval + 2))

    merge_cells(
        spreadsheet=sht1,
        worksheet=work_sheet,
        row_spans=merge_rspans,
        col_spans=merge_cspans
    )


if __name__ == '__main__':
    main()