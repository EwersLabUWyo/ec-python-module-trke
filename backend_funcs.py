import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import skopt
from pathlib import Path

import FluxFox
from FluxFox.ameriflux import retrieve_ameriflux
from FluxFox import postproc
from FluxFox import thermo



def load_amf_data(site_id, username, email, use_descr="Wyoming EPSCoR Teacher-Researcher Knowledge Exchange workshop."):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cache_dir = Path("./cache")
        cache_dir.mkdir(exist_ok=True)
        try:
            print(f"Attempting to locate cached raw Ameriflux data for site {site_id}...")
            with open(cache_dir / f"{site_id}_amf_raw.pickle", "rb") as f:
                amf = pickle.load(f)
            print("Success...")
        except FileNotFoundError:
            print(f"Could not find raw data cache. Downloading from Ameriflux servers...")
            amf = retrieve_ameriflux(
                site_id=site_id,
                username=username,
                email=email,
                intended_use="education",
                use_descr=use_descr
            )
            with open(cache_dir / f"{site_id}_amf_raw.pickle", "wb") as f:
                pickle.dump(amf, f)
    return amf

_INTEGRITY_GRADIENT_STEPS = 256
_LOW_MISSING_THRESHOLD = 0.05
_HIGH_MISSING_THRESHOLD = 0.95
_COLORBAR_TICKS = [0, _LOW_MISSING_THRESHOLD, 0.25, 0.5, 0.75, _HIGH_MISSING_THRESHOLD, 1]
_COLORBAR_LABELS = ["0%", "5%", "25%", "50%", "75%", "95%", "100%"]


def _validate_integrity_dataframe(df):
    """Validate dataframe requirements for integrity visualization."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a pandas DatetimeIndex.")
    if df.empty:
        raise ValueError("df must contain at least one row.")
    if df.index.hasnans:
        raise ValueError("df index must not contain NaT values.")
    if not df.columns.is_unique:
        raise ValueError("df column names must be unique.")


def _select_integrity_columns(df, columns, regex):
    """Return dataframe columns selected by explicit names and an optional regex."""
    if columns is None:
        selected_columns = list(df.columns)
    elif isinstance(columns, str):
        selected_columns = [columns]
    else:
        selected_columns = list(columns)

    missing_columns = [column for column in selected_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Columns not found in df: {missing_columns}")

    if regex is not None:
        try:
            regex_columns = set(df.filter(regex=regex).columns)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid column-selection regex {regex!r}.") from error
        selected_columns = [
            column for column in selected_columns if column in regex_columns
        ]

    if not selected_columns:
        raise ValueError("Column selection did not match any dataframe columns.")
    return selected_columns


def _build_integrity_colormap(cmap, low_missing_color, high_missing_color):
    """Create the three-zone colormap used by the missingness heatmap."""
    middle_colormap = plt.get_cmap(cmap)
    middle_positions = np.linspace(0, 1, _INTEGRITY_GRADIENT_STEPS)
    middle_range = _HIGH_MISSING_THRESHOLD - _LOW_MISSING_THRESHOLD
    # Adjacent stops create sharp transitions into the solid endpoint bands.
    color_stops = [
        (0.0, low_missing_color),
        (np.nextafter(_LOW_MISSING_THRESHOLD, 0), low_missing_color),
        *[
            (
                _LOW_MISSING_THRESHOLD + middle_range * position,
                middle_colormap(position),
            )
            for position in middle_positions
        ],
        (np.nextafter(_HIGH_MISSING_THRESHOLD, 1), high_missing_color),
        (1.0, high_missing_color),
    ]
    return LinearSegmentedColormap.from_list("data_integrity", color_stops)


def plot_data_integrity(
    df,
    columns=None,
    regex=None,
    time_interval="ME",
    figsize=None,
    cmap="cool",
    low_missing_color="black",
    high_missing_color="white",
    title=None,
):
    """Visualize variable availability and missingness over time.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with a DatetimeIndex.
    columns : sequence of str, optional
        Explicit column names to inspect. When combined with ``regex``, only
        explicitly named columns that also match the expression are included.
    regex : str, optional
        Regular expression used to select columns, such as ``r"^FC"``.
    time_interval : str
        Pandas resampling frequency for the missingness heatmap.
    figsize : tuple of float, optional
        Figure dimensions in inches. A height based on the number of selected
        columns is used when omitted.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap used between 5% and 95% missingness.
    low_missing_color : color, optional
        Color for periods with less than 5% missingness.
    high_missing_color : color, optional
        Color for periods with more than 95% missingness.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The data-integrity heatmap.
    """
    _validate_integrity_dataframe(df)
    selected_columns = _select_integrity_columns(df, columns, regex)
    selected = df.loc[:, selected_columns].sort_index()
    try:
        # Empty resampling bins represent periods with no observations at all.
        missing_by_period = selected.isna().resample(time_interval).mean().fillna(1.0)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"time_interval {time_interval!r} is not a valid pandas frequency."
        ) from error

    availability = (1 - selected.isna().mean()).sort_values()

    if figsize is None:
        figsize = (16, max(8, 3 + 0.28 * len(selected_columns)))

    figure, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    if title is not None:
        figure.suptitle(title, fontsize=16)

    availability_percent = availability * 100
    axes[0].barh(availability_percent.index, availability_percent, color="tab:blue")
    axes[0].invert_yaxis()
    axes[0].set(
        title="Data availability by variable",
        xlabel="Available observations (%)",
        xlim=(0, 100),
    )

    integrity_colormap = _build_integrity_colormap(
        cmap,
        low_missing_color,
        high_missing_color,
    )
    heatmap = axes[1].imshow(
        missing_by_period.T.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=integrity_colormap,
        vmin=0,
        vmax=1,
    )
    heatmap_positions = np.linspace(
        0, len(missing_by_period.index) - 1, min(8, len(missing_by_period.index)), dtype=int
    )
    axes[1].set(
        title=f"Missingness by {time_interval} period",
        ylabel="Variable",
        xticks=heatmap_positions,
        xticklabels=[
            missing_by_period.index[position].strftime("%Y-%m-%d")
            for position in heatmap_positions
        ],
        yticks=np.arange(len(selected_columns)),
        yticklabels=selected_columns,
    )
    axes[1].tick_params(axis="x", rotation=45)
    colorbar = figure.colorbar(
        heatmap,
        ax=axes[1],
        ticks=_COLORBAR_TICKS,
        label="Missing observations",
    )
    colorbar.minorticks_off()
    colorbar.ax.set_yticklabels(_COLORBAR_LABELS)
    return figure

# Run this cell
# It can take 3-15 minutes to run, depending on the length of the data.


def postprocess(amf, LE_var, FC_var, H_var, P_var, USTAR_var):

    print("Performing Post-Processing and Quality Control Routines. This could take several minutes.")

    rh_cols = amf.data.filter(regex="^RH(_PI|_F|_[0-9]+_1|$)").columns
    rh_dat = amf.data[rh_cols].mean(1).rename("RH")
    swc_cols = amf.data.filter(regex="^SWC(_PI|_F|_[0-9]+_1|$)").columns
    swc_dat = amf.data[swc_cols].mean(1).rename("SWC")
    swin_cols = amf.data.filter(regex="^SW_IN(_PI|_F|_[0-9]+_1|$)").columns
    swin_dat = amf.data[swin_cols].mean(1).rename("SW_IN")
    ta_cols = amf.data.filter(regex="^TA(_PI|_F|_[0-9]+_1|$)").columns
    ta_dat = amf.data[ta_cols].mean(1).rename("TA")
    vpd_dat = thermo.compute_e_sat_water(ta_dat, check_inputs=False) * (1 - rh_dat/100)
    vpd_dat.name = "VPD"
    lw_cols = amf.data.filter(regex="^LW_OUT").columns
    lw_dat = amf.data[lw_cols].mean(1).rename("LW_OUT")

    df = amf.data[[LE_var, FC_var, H_var, P_var, USTAR_var]].copy()

    df = df.rename(columns={LE_var: "LE", FC_var: "NEE", P_var: "P", USTAR_var: "USTAR", H_var:"H"})
    df["RH"] = rh_dat
    df["SWC"] = swc_dat
    df["SW_IN"] = swin_dat
    df["TA"] = ta_dat
    df["VPD"] = vpd_dat
    df["LW_OUT"] = lw_dat

    print("(1/4) Computing sunrise/sunset times...", end="")
    isday = postproc.compute_isday(df.index, lat=amf.lat, lon=amf.lon, elev=amf.elev)
    print("Done")

    print("(2/4) Despiking data...", end="")
    cols_to_despike = ["LE", "NEE", "H", "TA", "SW_IN"]
    keep = postproc.despike_mad_papale_2006(df, isday, cols_to_despike=cols_to_despike)
    pct_integrity = (keep.count() / keep.shape[0] * 100).round(2)
    df.loc[:, cols_to_despike] = df.loc[:, cols_to_despike].where(keep)
    new_integrity = (df[cols_to_despike].notna().sum() / df.shape[0] * 100).round(2)
    print(f"Done (Integrity before: {pct_integrity.to_dict()}, after: {new_integrity.to_dict()})")
    
    print("(3/4) Removing time periods with insufficient wind...", end="")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ust_res = postproc.ustar_broken_stick(
            df, isday, 
            ta_col="TA", ustar_col="USTAR", nee_col="NEE", 
            seasons=3, n_ta_classes=6, ustar_ta_corr_cutoff=0.4, ustar_thresh_bounds=(0.1, 0.7), n_restarts=10, 
            loss="mae", gapfill_quantile=0.75, default_ustar_thresh=0.2, pbar=False, random_state=8472
        )
    df.loc[:, ["LE", "NEE", "H"]] = df.loc[:, ["LE", "NEE", "H"]].where(ust_res.flag)
    print("Done")
    print(ust_res.thresholds)
    print("Percent of values removed by ustar filter:", ((1 - ust_res.flag.mean())*100).round(2))


    print("(4/4) Filling in data gaps (this is the slowest step)...", end="")
    for tgt_col in ["LE", "NEE", "H"]:
        if df[tgt_col].isna().all():
            df[f"{tgt_col}_xgb"] = np.nan
            df[f"{tgt_col}_xgb_mds"] = np.nan
            continue
            # only for now
        xgb_res = postproc.gapfill_xgb_liu_2025(
            df, tgt_col,
            ta_col="TA", sw_in_col="SW_IN", swc_col="SWC", vpd_col="VPD", extra_predictor_cols=["RH"],
            n_missing_allowed=1, 
            hyper_train_frac=min(0.2, 17500/df.shape[0]),
            hyper_test_frac=min(0.07, min(0.07, 17500/df.shape[0]*0.3)),
            train_frac = min(0.8, 17500*5/df.shape[0]),
            n_bayes_iter = 30, cv_folds=5, random_state=8472, verbose=False
        )
        print(".", end="")
        df[f"{tgt_col}_xgb"] = xgb_res.filled

        mds_res = postproc.gapfill_mds_reichstein_2005(
            df, f"{tgt_col}_xgb",
            ta_col="TA", sw_in_col="SW_IN", vpd_col="VPD", tol_vpd=50, verbose=False
        )
        print(".", end="")
        df[f"{tgt_col}_xgb_mds"] = mds_res.filled
    print("Done")
    print("All post-processing and quality control procedures finished.")
    df[["WY", "DOWY"]] = FluxFox.utils.compute_WY_DOWY(df.index)
    return df

__all__ = ["plot_data_integrity", "load_amf_data", "postprocess", "_process_site"]


def _process_site(site_id, username, email, var_map, years=None):
    """Load, postprocess, and save a single site. Runs in a worker process."""
    amf = load_amf_data(site_id, username, email)
    if years is not None:
        amf.data = amf.data.loc[years[0]:years[1]]
    post_df = postprocess(amf=amf, **var_map)
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    post_df.to_parquet(out_dir / f"{site_id}.parquet")
    return site_id

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, as_completed

    username = "afox18"  # change me to your ameriflux username
    email = "afox18@uwyo.edu"  # change me to the email associated with your ameriflux account

    sites = {
        "CA-Obs": dict(
            years=("2000", "2009"),
            var_map=dict(LE_var="LE", FC_var="FC", H_var="H", P_var="P", USTAR_var="USTAR"),
        ),
        "US-Kon": dict(
            years=None,
            var_map=dict(
                LE_var="LE_1_1_1", FC_var="FC_1_1_1", H_var="H_1_1_1",
                P_var="P_1_1_1", USTAR_var="USTAR_1_1_1",
            ),
        ),
        "US-GLE": dict(
            years=("2010", "2020"),
            var_map=dict(
                LE_var="LE_PI_F_1_1_1", FC_var="FC_1_1_1", H_var="H_PI_F_1_1_1",
                P_var="P_PI_F", USTAR_var="USTAR_PI_F_1_1_1",
            ),
        ),
    }

    with ProcessPoolExecutor(max_workers=len(sites)) as pool:
        futures = {
            pool.submit(_process_site, site_id, username, email, **cfg): site_id
            for site_id, cfg in sites.items()
        }
        for fut in as_completed(futures):
            print(f"\nFinished {fut.result()}")
