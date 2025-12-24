# prepare_panel.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import glob
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


_SYMBOL_RE = re.compile(r"^(\d{6}\.(?:SH|SZ))_day_qfq\.csv$", re.IGNORECASE)


def discover_csvs(data_dir: str, pattern: str = "*_day_qfq.csv") -> List[str]:
    paths = glob.glob(os.path.join(data_dir, pattern))
    paths = [p for p in paths if os.path.isfile(p)]
    paths.sort()
    return paths


def symbol_from_filename(path: str) -> Optional[str]:
    name = os.path.basename(path)
    m = _SYMBOL_RE.match(name)
    if not m:
        return None
    return m.group(1).upper()


def read_price_table(path: str) -> pd.DataFrame:
    """
    Robust CSV/TSV loader:
    - Try comma-separated first
    - If essential columns missing, retry with tab-separated
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # If it is actually TSV but saved as .csv, comma parsing often yields very few columns.
    if ("trade_date" not in df.columns) and ("date" not in df.columns) and (df.shape[1] <= 2):
        df = pd.read_csv(path, sep="\t")
        df.columns = [c.strip() for c in df.columns]

    return df


def _infer_date_col(cols: List[str]) -> str:
    for c in ["trade_date", "date", "datetime", "time"]:
        if c in cols:
            return c
    return cols[0]


def _parse_dates(s: pd.Series) -> pd.DatetimeIndex:
    # handles YYYY-MM-DD, YYYY/MM/DD, YYYYMMDD, numeric yyyymmdd
    if np.issubdtype(s.dtype, np.number):
        s = s.astype("int64").astype(str)
    s = s.astype(str).str.strip()

    if s.str.fullmatch(r"\d{8}").all():
        return pd.to_datetime(s, format="%Y%m%d", errors="raise")
    return pd.to_datetime(s, errors="coerce")


def load_panel(
    data_dir: str,
    fields: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_files: Optional[int] = None,
    dtype: str = "float64",
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Returns:
      panel: Dict[field, DataFrame(T,N)]
        index   : pd.DatetimeIndex (T,)
        columns : pd.Index[str]     (N,)
        values  : float{32|64}      (T,N)
    """
    paths = discover_csvs(data_dir)
    if max_files is not None:
        paths = paths[:max_files]

    base_field = "close" if "close" in fields else fields[0]

    series_map: Dict[str, List[pd.Series]] = {f: [] for f in fields}
    kept, skipped = 0, 0

    for p in paths:
        sym = symbol_from_filename(p)
        if sym is None:
            skipped += 1
            continue

        try:
            df = read_price_table(p)

            # prefer symbol from file content if available
            if "ts_code" in df.columns:
                uniq = df["ts_code"].dropna().astype(str).str.strip().unique()
                if len(uniq) == 1:
                    sym = uniq[0].upper()

            df.columns = [c.strip() for c in df.columns]
            cols = list(df.columns)

            dcol = _infer_date_col(cols)
            dt = _parse_dates(df[dcol])
            if pd.isna(dt).all():
                raise ValueError(f"Cannot parse dates from column '{dcol}'")

            df2 = df.copy()
            df2["__dt"] = dt
            df2 = df2.dropna(subset=["__dt"]).set_index("__dt").sort_index()
            df2 = df2[~df2.index.duplicated(keep="last")]

            if base_field not in df2.columns:
                raise ValueError(f"Missing base_field '{base_field}'")

            for f in fields:
                if f not in df2.columns:
                    continue
                s = pd.to_numeric(df2[f], errors="coerce")
                s = pd.Series(s.values, index=df2.index, name=sym)
                series_map[f].append(s)

            kept += 1

        except Exception as e:
            skipped += 1
            if verbose:
                print(f"[WARN] skip {os.path.basename(p)}: {e}")

    if kept == 0 or len(series_map.get(base_field, [])) == 0:
        raise RuntimeError(f"No valid files loaded from {data_dir} for base_field='{base_field}'")

    base_mat = pd.concat(series_map[base_field], axis=1).sort_index()

    if start is not None:
        base_mat = base_mat.loc[base_mat.index >= pd.to_datetime(start)]
    if end is not None:
        base_mat = base_mat.loc[base_mat.index <= pd.to_datetime(end)]

    min_obs = max(50, int(0.05 * len(base_mat)))
    base_mat = base_mat.dropna(axis=1, thresh=min_obs)

    dates = base_mat.index
    symbols = base_mat.columns

    panel: Dict[str, pd.DataFrame] = {base_field: base_mat}

    for f in fields:
        if f == base_field:
            continue
        if len(series_map[f]) == 0:
            panel[f] = pd.DataFrame(np.nan, index=dates, columns=symbols)
            continue

        mat = pd.concat(series_map[f], axis=1).sort_index()

        if start is not None:
            mat = mat.loc[mat.index >= pd.to_datetime(start)]
        if end is not None:
            mat = mat.loc[mat.index <= pd.to_datetime(end)]

        panel[f] = mat.reindex(index=dates, columns=symbols)

    # dtype cast
    np_dtype = np.float32 if dtype.lower() == "float32" else np.float64
    for f in panel.keys():
        panel[f] = panel[f].astype(np_dtype)

    if verbose:
        print(f"[INFO] loaded files={kept}, skipped_files={skipped}")
        print(f"[INFO] panel fields={list(panel.keys())}")
        print(f"[INFO] dates={len(dates)}, symbols={len(symbols)}, dtype={np_dtype}")

    return panel


def stack_panel_to_3d(panel: Dict[str, pd.DataFrame], fields: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    panel[field]: DataFrame(T,N) with identical index/columns across fields.
    returns:
      data   : ndarray(T,N,F)
      dates  : ndarray(T,) datetime64[ns]
      symbols: ndarray(N,) str
      fields : ndarray(F,) str
    """
    base = panel[fields[0]]
    dates_pd = base.index
    symbols_pd = base.columns

    for f in fields[1:]:
        df = panel[f]
        if (not df.index.equals(dates_pd)) or (not df.columns.equals(symbols_pd)):
            raise ValueError(f"Field '{f}' is not aligned with '{fields[0]}'")

    data = np.stack([panel[f].to_numpy(copy=False) for f in fields], axis=2)

    dates = dates_pd.values.astype("datetime64[ns]")
    symbols = symbols_pd.to_numpy(dtype=str)
    fields_arr = np.array(fields, dtype=str)

    return data, dates, symbols, fields_arr


def save_panel_npz(
    out_path: str,
    data: np.ndarray,
    dates: np.ndarray,
    symbols: np.ndarray,
    fields: np.ndarray,
    compress: bool = True,
) -> None:
    """
    Save a panel as an npz:
      - data: (T,N,F)
      - dates: (T,) datetime64[ns]
      - symbols: (N,) str
      - fields: (F,) str
    """
    if compress or out_path.endswith(".npz"):
        np.savez_compressed(out_path, data=data, dates=dates, symbols=symbols, fields=fields)
    else:
        np.savez(out_path, data=data, dates=dates, symbols=symbols, fields=fields)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--fields", type=str, default="open,high,low,close,volume,amount,pre_close")
    ap.add_argument("--start", type=str, default="")
    ap.add_argument("--end", type=str, default="")
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])
    ap.add_argument("--out_path", type=str, default="panel.npz")
    ap.add_argument("--compress", type=int, default=1)  # 1/0
    args = ap.parse_args()

    fields = [x.strip() for x in args.fields.split(",") if x.strip()]
    start = args.start or None
    end = args.end or None
    max_files = args.max_files if args.max_files and args.max_files > 0 else None
    compress = bool(args.compress)

    panel = load_panel(
        data_dir=args.data_dir,
        fields=fields,
        start=start,
        end=end,
        max_files=max_files,
        dtype=args.dtype,
        verbose=True,
    )

    data, dates_np, symbols_np, fields_np = stack_panel_to_3d(panel, fields)
    save_panel_npz(args.out_path, data, dates_np, symbols_np, fields_np, compress=compress)

    print(f"[INFO] saved npz to: {args.out_path}")
    print(f"[INFO] data shape: {data.shape} (T,N,F)")


if __name__ == "__main__":
    main()
