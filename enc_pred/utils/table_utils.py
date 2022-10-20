"""
This is to define a table class that is able to apply result to table.
"""
from typing import *


def apply_effect(text: Union[Text, float], effect: Optional[Text]) -> Text:
    """Apply effect according to effect specification
    """
    return f"{effect} {text}" if effect is not None and text != '' else str(text)


class Cell:
    """
    This is the base cell class that is used to apply 
    effect to items.
    """
    def __init__(
        self,
        text: Union[Text, float],
        rows: Optional[int] = 1,
        columns: Optional[int] = 1,
        align: Optional[Text] = 'l',
        style: Optional[Text] = None,
        effect: Optional[Text] = None,
        num_digit: int = 2,
        signed: bool = False
    ):
        self._text = text
        self._effect = effect
        self._columns = columns
        self._align = align
        self._style = style
        self._rows = rows
        self._num_digit = num_digit
        self._signed = signed

    @property
    def text(self) -> Text:
        """
        This is the way that we apply a wrapped text
        from the original source.
        """
        stl_wrap = lambda x: f'\\text{self._style}{{{x}}}' if self._style is not None and x != '' else x
        mr_wrap = lambda x: f'\\multirow{{{self._rows}}}{{*}}{{{x}}}' if self._rows != 1 else x
        mc_wrap = lambda x: f'\\multicolumn{{{self._columns}}}{{{self._align}}}{{{x}}}' if self._columns != 1 else x

        text = self._text
        if isinstance(text, float):
            if self._signed:
                text = f'+{text * 100:.{self._num_digit}f}' if text > 0 else f'-{abs(text) * 100:.{self._num_digit}f}'
            else:
                text = f'{text * 100:.{self._num_digit}f}'

        return mc_wrap(mr_wrap(stl_wrap(text)))

    def set_text(self, text: Text):
        """
        Explicitely use a different interface to differentiate
        from the text displayed by property 'text'
        """
        self._text = text

    @property
    def effect(self) -> Union[Text, None]:
        """
        examine the effect of this item
        """
        return self._effect

    @effect.setter
    def effect(self, effect: Text):
        self._effect = effect

    def apply_effect(self) -> Text:
        """
        Apply the effect to text for generating
        the actual cell content.
        """

        # use the text property instead of raw text
        return apply_effect(text=self.text, effect=self._effect)


class Row:
    """
    This is the data structure that 
    host a list of cells.
    """
    def __init__(
        self,
        cells: Optional[List[Cell]] = None,
        before_row: Optional[Text] = None,
        after_row: Optional[Text] = None
    ):
        """
        """
        self._cells = cells if cells is not None else []
        self._before_row = before_row
        self._after_row = after_row

    def append(self, cell: Cell):
        """
        Append cell to then end of the cell list.
        """
        self._cells.append(cell)

    @property
    def after_row(self) -> Union[Text, None]:
        """
        """
        return self._after_row

    @after_row.setter
    def after_row(self, after_row: Text):
        self._after_row = after_row

    def render_cells(self) -> Text:
        """
        Render the cells and generate a cell list.
        """
        return_stem = ' & '.join(
            [cell.apply_effect() for cell in self._cells]
        )

        before_row = f'{self._before_row}\n' if self._before_row is not None else ''
        after_row = f'{self._after_row}\n' if self._after_row is not None else ''

        return f'{before_row}{return_stem}\\\\\n{after_row}'

    def __getitem__(self, x: int):
        return self._cells[x]

    def __len__(self, x: int):
        return len(self._cells)


class MyTable:
    """
    This is the table class that holds a list of rows.
    """
    def __init__(
        self,
        header: Text,
        tail: Text,
        rows: Optional[List[Row]] = None
    ):
        """
        """
        self._header = header
        self._tail = tail
        self._rows = rows if rows is not None else []

    def render_rows(self) -> Text:
        """
        Generate rendered rows by combining rows.
        """

        table_stem = ''.join([row.render_cells() for row in self._rows])
        return f'{self._header}\n{table_stem}\n{self._tail}'

    def insert_row(self, row: Row):
        """
        """
        self._rows.append(row)

    def __getitem__(self, x: int):
        """This function returns the x position
        item in the rows
        """
        return self._rows[x]

    def __len__(self) -> int:
        return len(self._rows)
