"""Excel and CSV loading as markdown tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SheetContent:
    page_num: int
    text: str
    source_type: str = "excel"


def _df_to_markdown(df: pd.DataFrame) -> str:
    df = df.fillna("")
    headers = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join([headers, sep, *rows])


def load_csv(path: Path) -> list[SheetContent]:
    df = pd.read_csv(path)
    return [SheetContent(page_num=1, text=_df_to_markdown(df))]


def load_excel(path: Path) -> list[SheetContent]:
    xl = pd.ExcelFile(path)
    sheets: list[SheetContent] = []
    for idx, name in enumerate(xl.sheet_names, start=1):
        df = pd.read_excel(path, sheet_name=name)
        header = f"Sheet: {name}\n\n"
        sheets.append(
            SheetContent(page_num=idx, text=header + _df_to_markdown(df))
        )
    return sheets
