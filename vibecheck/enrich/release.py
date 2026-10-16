"""Best guess at when a recording first came out.

Album release dates lie for remasters and compilations ("Greatest Hits", 2015), so we also read
the year code inside the ISRC (CC-XXX-YY-NNNNN, YY = year the code was assigned). Reissues normally
keep the original ISRC, so the earlier of the two years is usually the original release.
"""
from __future__ import annotations

import datetime as _dt
import re

import numpy as np
import pandas as pd


def isrc_year(isrc, now_year: int | None = None) -> int | None:
    if not isinstance(isrc, str):
        return None
    code = re.sub(r"[^A-Za-z0-9]", "", isrc).upper()
    if len(code) != 12 or not code[5:7].isdigit():
        return None
    yy = int(code[5:7])
    now_year = now_year or _dt.date.today().year
    year = 2000 + yy if 2000 + yy <= now_year else 1900 + yy
    return year if year >= 1940 else None  # ISRCs date from 1986; pre-1940 codes are junk


def album_year(release_date) -> int | None:
    if release_date is None or (isinstance(release_date, float) and np.isnan(release_date)):
        return None
    m = re.match(r"\s*(\d{4})", str(release_date))
    return int(m.group(1)) if m and 1900 <= int(m.group(1)) <= _dt.date.today().year + 1 else None


def best_year(release_date, isrc) -> int | None:
    years = [y for y in (album_year(release_date), isrc_year(isrc)) if y]
    return min(years) if years else None


def add_release_years(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rd = df["release_date"] if "release_date" in df else pd.Series([None] * len(df), index=df.index)
    isrc = df["isrc"] if "isrc" in df else pd.Series([None] * len(df), index=df.index)
    df["release_year"] = [best_year(r, i) for r, i in zip(rd, isrc)]
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")
    return df
