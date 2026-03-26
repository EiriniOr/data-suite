import pandas as pd


def detect_time_series(df: pd.DataFrame) -> dict:
    """Detect whether a DataFrame is a time series dataset.

    Returns a dict with:
        is_ts: bool
        datetime_col: str | None
        freq: str | None  (e.g. 'D', 'W', 'M', 'H')
        n_periods: int
        notes: str
    """
    result = {
        "is_ts": False,
        "datetime_col": None,
        "freq": None,
        "n_periods": 0,
        "notes": "",
    }

    # ── Find datetime columns ─────────────────────────────────────────────────
    datetime_cols = [
        c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
    ]

    # Try to parse object/string columns as datetime
    if not datetime_cols:
        for col in df.select_dtypes(include=["object"]).columns:
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True)
                if parsed.notna().sum() / len(df) > 0.8:
                    datetime_cols.append(col)
            except Exception:
                continue

    if not datetime_cols:
        result["notes"] = "No datetime column detected."
        return result

    # Pick the most complete datetime column
    dt_col = max(datetime_cols, key=lambda c: df[c].notna().sum())
    result["datetime_col"] = dt_col

    # ── Check if it's monotonic (ordered time series) ─────────────────────────
    try:
        parsed = pd.to_datetime(df[dt_col])
        if parsed.is_monotonic_increasing or parsed.is_monotonic_decreasing:
            result["is_ts"] = True
        else:
            result["notes"] = (
                f"`{dt_col}` is datetime but not monotonic — may not be a time series."
            )
            return result
    except Exception:
        result["notes"] = f"Could not parse `{dt_col}` as datetime."
        return result

    # ── Infer frequency ───────────────────────────────────────────────────────
    try:
        inferred = pd.infer_freq(parsed.dropna())
        result["freq"] = inferred or _estimate_freq(parsed)
    except Exception:
        result["freq"] = _estimate_freq(parsed)

    result["n_periods"] = parsed.notna().sum()
    result["notes"] = (
        f"Time series detected via `{dt_col}`. "
        f"Frequency: {result['freq'] or 'unknown'}, {result['n_periods']} periods."
    )
    return result


def _estimate_freq(series: pd.Series) -> str | None:
    """Fallback: estimate frequency from median gap between timestamps."""
    deltas = series.dropna().sort_values().diff().dropna()
    if deltas.empty:
        return None
    median_delta = deltas.median()
    seconds = median_delta.total_seconds()

    if seconds < 3600:
        return "T"  # minute-ish
    elif seconds < 86400:
        return "H"  # hourly
    elif seconds < 86400 * 4:
        return "D"  # daily
    elif seconds < 86400 * 10:
        return "W"  # weekly
    elif seconds < 86400 * 35:
        return "M"  # monthly
    elif seconds < 86400 * 100:
        return "Q"  # quarterly
    else:
        return "A"  # annual
