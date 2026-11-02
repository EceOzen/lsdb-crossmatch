"""Run nway on N catalogs for a single aligned HEALPix pixel.

This is the "calling nway" building block of the N-way crossmatch
(astronomy-commons/lsdb-crossmatch#32). It knows nothing about alignment or
Dask: given the partition data that falls in one aligned pixel, it builds the
nway input tables, calls ``nwaylib.nway_match`` and returns per-catalog row
indices plus nway's probability columns. The plumbing that iterates over
aligned pixels lives in #31.

Current assumptions (as described in #32, to be revisited later):

- Source densities are computed per pixel: every catalog's sky area is the
  area of the aligned pixel.
- Magnitude histograms are built automatically by nway from the rows in the
  pixel, not from the whole catalog.
  
  --> Linh's update from Ece's PR #33:
  
  This PR extends the N-way pixel-level crossmatching implementation to support global magnitude histograms.
  
  - Collect magnitude samples across pixels.
  - Estimate global magnitude histograms from those samples.
  - Accept precomputed global histograms.
  - Pass the global histograms to nway during pixel-level matching.
"""
  #Local tests have not yet been run because of some local setup issues. Runtime validation still needed to be done.


from __future__ import annotations

from dataclasses import dataclass, field

import hats.pixel_math.healpix_shim as hp
import numpy as np
import pandas as pd
from hats.pixel_math import HealpixPixel
from lsdb.operations.functions.merge_catalog_functions import filter_by_spatial_index_to_pixel
from nwaylib import nway_match
from nwaylib import magnitudeweights

NO_COUNTERPART = -1


@dataclass(frozen=True)
class NwayCatalogConfig:
    """Describes how to read one catalog's partition for nway.

    Parameters
    ----------
    name : str
        Short catalog name with no spaces. nway uses it in output column names,
        so names must be unique across the catalogs in one crossmatch.
    ra_column, dec_column : str
        Position columns, in degrees.
    pos_err_column : str, optional
        Circular positional error, in arcsec.
    ra_err_column, dec_err_column : str, optional
        Per-axis positional errors, in arcsec. Used when ``pos_err_column`` is
        not given; combined as ``sqrt((ra_err * cos(dec))**2 + dec_err**2)``,
        matching the existing two-catalog ``NWAYCrossmatch``.
    mag_columns : list of str, optional
        Columns used as magnitude priors.
    """

    name: str
    ra_column: str
    dec_column: str
    pos_err_column: str | None = None
    ra_err_column: str | None = None
    dec_err_column: str | None = None
    mag_columns: list[str] = field(default_factory=list)

    def validate(self, columns) -> None:
        """Raise ``ValueError`` if a configured column is missing from ``columns``."""
        if " " in self.name:
            raise ValueError(f"Catalog name '{self.name}' must not contain spaces.")
        if self.pos_err_column is None and (self.ra_err_column is None or self.dec_err_column is None):
            raise ValueError(
                f"Catalog '{self.name}': give either pos_err_column or both ra_err_column and dec_err_column."
            )
        wanted = [self.ra_column, self.dec_column, *self.mag_columns]
        if self.pos_err_column is not None:
            wanted.append(self.pos_err_column)
        else:
            wanted += [self.ra_err_column, self.dec_err_column]
        missing = [col for col in wanted if col not in columns]
        if missing:
            raise ValueError(f"Catalog '{self.name}' is missing columns: {missing}")

    def positional_errors(self, df: pd.DataFrame) -> np.ndarray:
        """Positional error per row, in arcsec."""
        if self.pos_err_column is not None:
            return df[self.pos_err_column].to_numpy(dtype=float)
        ra_err = df[self.ra_err_column].to_numpy(dtype=float)
        dec_err = df[self.dec_err_column].to_numpy(dtype=float)
        dec_rad = np.radians(df[self.dec_column].to_numpy(dtype=float))
        return np.sqrt((ra_err * np.cos(dec_rad)) ** 2 + dec_err**2)


def pixel_area_deg2(order: int) -> float:
    """Area of one HEALPix pixel at ``order``, in square degrees (the unit nway expects)."""
    return float(hp.order2pixarea(order, degrees=True))


def prepare_primary(primary_df: pd.DataFrame, primary_pixel: HealpixPixel, aligned_pixel: HealpixPixel):
    """Keep only the primary rows that fall inside the aligned pixel.

    The aligned pixel can be smaller than the primary's partition pixel, in
    which case only part of the partition belongs to this pixel.
    """
    if aligned_pixel.order <= primary_pixel.order:
        return primary_df
    return filter_by_spatial_index_to_pixel(primary_df, aligned_pixel.order, aligned_pixel.pixel)


def prepare_secondary(df: pd.DataFrame | None, margin_df: pd.DataFrame | None) -> pd.DataFrame:
    """Combine a secondary catalog's partition with its margin.

    Counterparts of primary sources near the pixel edge can sit just outside
    the pixel, so secondaries are matched against partition + margin.
    """
    frames = [frame for frame in (df, margin_df) if frame is not None and len(frame) > 0]
    if not frames:
        return df if df is not None else margin_df
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames)


def build_nway_table(
    df: pd.DataFrame,
    config: NwayCatalogConfig,
    area_deg2: float,
    maghists: list | None = None #allows NWAY's support for optional precomputed magnitude histograms, currently automatic calc is hard-coded
    ) -> dict:
    """Build one entry of nway's ``match_tables`` list."""
    # nway overwrites -99 with NaN in place, so give it its own float copies.
    mags = [df[col].to_numpy(dtype=float, copy=True) for col in config.mag_columns]

    #if no histogram provided, nway automatically estimate mag priors
    if maghists is None:
        maghists = [None] * len(mags)

    if len(maghists) != len(mags):
        raise ValueError(
            f"Expected {len(mags)} magnitude histograms, "
            f"but received {len(maghists)}."
        )

    return {
        "name": config.name,
        "ra": df[config.ra_column].to_numpy(dtype=float),
        "dec": df[config.dec_column].to_numpy(dtype=float),
        "error": config.positional_errors(df),
        "area": area_deg2,
        "mags": mags,
        "magnames": list(config.mag_columns),
        # One entry per magnitude column; None means "let nway build the
        # histogram". An empty list would make nway silently skip the priors.
        "maghists": maghists,
    }

def build_positional_nway_table(
    df: pd.DataFrame,
    config: NwayCatalogConfig,
    area_deg2: float,
) -> dict: #build nway table w/out mag info

    return {
        "name": config.name,
        "ra": df[config.ra_column].to_numpy(dtype=float),
        "dec": df[config.dec_column].to_numpy(dtype=float),
        "error": config.positional_errors(df),
        "area": area_deg2,
        "mags": [],
        "magnames": [],
        "maghists": [],
    }

def nway_crossmatch_pixel_positional(
    frames: list[pd.DataFrame],
    configs: list[NwayCatalogConfig],
    aligned_order: int,
    *,
    radius_arcsec: float,
    prior_completeness: float | list[float] = 1.0,
) -> tuple[list[np.ndarray], pd.DataFrame]:

    #validate the inputs
    if len(frames) != len(configs):
        raise ValueError("Number of frames and configs must match.")

    if len(frames) < 2:
        raise ValueError("nway needs at least two catalogs.")

    names = [config.name for config in configs]

    for frame, config in zip(frames, configs):
        config.validate(frame.columns)

    #return if primary or secondary is empty
    if len(frames[0]) == 0 or all(len(frame) == 0 for frame in frames[1:]):
        return _empty_result(names)

    area = pixel_area_deg2(aligned_order)

    tables = [
        build_positional_nway_table(frame, config, area)
        for frame, config in zip(frames, configs)
    ]

    results = nway_match(
        tables,
        radius_arcsec,
        prior_completeness,
        store_mag_hists=False,
    )

    return _clean_results(
        results,
        names,
        match_flag=0,
        keep_unmatched=True,
    )

def create_global_mag_samples(
    configs: list[NwayCatalogConfig],
) -> dict: #create empty containers to collect mag samples
    samples = {}

    for config in configs:
        samples[config.name] = {}

        for mag_column in config.mag_columns:
            samples[config.name][mag_column] = {
                "selected": [],
                "selected_weights": [],
                "background": [],
            }

    return samples

def collect_pixel_mag_samples(
    frames: list[pd.DataFrame],
    configs: list[NwayCatalogConfig],
    indices: list[np.ndarray],
    extra: pd.DataFrame,
    samples: dict,
) -> None: #collect mag samples from one-positional xmatch.

    #dealing with empty pixels
    if extra.empty or "dist_post" not in extra.columns:
        return
    if not (len(frames) == len(configs) == len(indices)):
        raise ValueError("Frames, configs and indices must have equal lengths.")

    dist_post = extra["dist_post"].to_numpy(dtype=float)

    for frame, config, catalog_indices in zip(frames, configs, indices):
        if len(catalog_indices) != len(dist_post):
            raise ValueError("Number of indices and probabilities must match.")
        #validate catelogue indices
        if np.any(catalog_indices < -1) or np.any(catalog_indices >= len(frame)):
            raise ValueError(
                f"Invalid catalogue indices for {config.name}."
            )
        #identify rows containing a source from this catalogue
        present = catalog_indices != -1
        selected_rows = (dist_post > 0.9) & present #select highly probable associations
        
        #keeping each catalogue source only once, with corresponding weight
        _, unique_positions = np.unique(
            catalog_indices[selected_rows],
            return_index=True,
        )

        selected_indices = catalog_indices[selected_rows][unique_positions]
        selected_weights = dist_post[selected_rows][unique_positions]

        #identify all sources appearing in possible associations
        possible_rows = (dist_post > 0.01) & present
        possible_indices = np.unique(catalog_indices[possible_rows])

        #catalogue sources not appearing in any plausible associations => background sources
        mask_background = np.ones(len(frame), dtype=bool)
        mask_background[possible_indices] = False

        for mag_column in config.mag_columns:
            magnitudes = frame[mag_column].to_numpy(dtype=float)
            mag_sel = magnitudes[selected_indices] #get selected mag & corresponding weights
            mag_all = magnitudes[mask_background] #get background mags

            #remove invalid mags
            valid_selected = np.isfinite(mag_sel) & (mag_sel != -99)
            valid_background = np.isfinite(mag_all) & (mag_all != -99)

            mag_sel = mag_sel[valid_selected]
            weights = selected_weights[valid_selected]
            mag_all = mag_all[valid_background]

            #append the pixel's samples to global collection
            column_samples = samples[config.name][mag_column]

            if len(mag_sel) > 0:
                column_samples["selected"].append(mag_sel)
                column_samples["selected_weights"].append(weights)

            if len(mag_all) > 0:
                column_samples["background"].append(mag_all)

def build_global_maghists(
    configs: list[NwayCatalogConfig],
    samples: dict,
) -> dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
#Build global magnitude histograms for every configured catalog column
    global_maghists = {}

    for config in configs:
        catalog_maghists = []

        for mag_column in config.mag_columns:
            column_samples = samples[config.name][mag_column]

            #check that samples were collected before concatenating
            if not column_samples["selected"]:
                raise ValueError(
                    f"No selected magnitude samples for "
                    f"{config.name}.{mag_column}."
                )
            
            if not column_samples["background"]:
                raise ValueError(
                    f"No background magnitude samples for "
                    f"{config.name}.{mag_column}."
                )

            #concat collected samples from all pixels
            mag_sel = np.concat(column_samples["selected"])
            weights = np.concat(column_samples["selected_weights"])
            mag_all = np.concat(column_samples["background"])

            #remove invalid mag values
            valid_selected = np.isfinite(mag_sel) & (mag_sel != -99)
            valid_background = np.isfinite(mag_all) & (mag_all != -99)

            mag_sel = mag_sel[valid_selected]
            weights = weights[valid_selected]
            mag_all = mag_all[valid_background]

            #check if global selected sample is large enough
            if len(mag_sel) < 100:
                raise ValueError(
                    f"Insufficient global selected magnitude samples for "
                    f"{config.name}.{mag_column}: {len(mag_sel)}."
                )

            #check if there's enough background sources
            if len(mag_all) == 0:
                raise ValueError(
                    f"No valid global background magnitude samples for "
                    f"{config.name}.{mag_column}."
                )

            #check selected source weights.
            if (
                len(weights) != len(mag_sel)
                or not np.all(np.isfinite(weights))
                or np.any(weights < 0)
                or np.sum(weights) <= 0
            ):
                raise ValueError(
                    f"Invalid selected-source weights for "
                    f"{config.name}.{mag_column}."
                )

            #build
            bins, hist_sel, hist_all = magnitudeweights.adaptive_histograms(
                mag_all,
                mag_sel,
                weights=weights,
            )

            catalog_maghists.append(
                (bins[:-1], bins[1:], hist_sel, hist_all)
            )

        global_maghists[config.name] = catalog_maghists

    return global_maghists


# pylint: disable=too-many-arguments
def nway_crossmatch_pixel(
    frames: list[pd.DataFrame],
    configs: list[NwayCatalogConfig],
    aligned_order: int,
    *,
    radius_arcsec: float,
    prior_completeness: float | list[float] = 1.0,
    global_maghists: dict | None = None, # allows for optional precomputed global mag histograms
    match_flag: int = 0,
    keep_unmatched: bool = False,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Run nway on N already-prepared frames from one aligned pixel.

    Parameters
    ----------
    frames : list of pd.DataFrame
        One frame per catalog, primary first. The primary should already be
        filtered to the aligned pixel (``prepare_primary``) and secondaries
        combined with their margins (``prepare_secondary``).
    configs : list of NwayCatalogConfig
        One config per frame, in the same order.
    aligned_order : int
        HEALPix order of the aligned pixel; its area is used for every catalog.
    radius_arcsec : float
        Maximum separation considered by nway.
    prior_completeness : float or list of float
        Passed to nway. A list needs one value per catalog, starting with 1.
    match_flag : {0, 1, 2}
        0 keeps every candidate combination, 1 only the best match per primary
        source, 2 the best match and nway's secondary alternatives.
    keep_unmatched : bool
        If False (default), drop rows with no counterpart in any secondary.

    Returns
    -------
    indices : list of np.ndarray
        One array per catalog with positional row indices into the matching
        frame (``frames[k].iloc[indices[k]]``), ``-1`` where that catalog has
        no counterpart in the row.
    extra : pd.DataFrame
        nway's per-row columns (separations, ``ncat``, ``match_flag``,
        ``prob_has_match`` (CLI ``p_any``), ``prob_this_match`` (CLI ``p_i``), ...).
    """
    if len(frames) != len(configs):
        raise ValueError(f"Got {len(frames)} frames but {len(configs)} configs.")
    if len(frames) < 2:
        raise ValueError("nway needs at least two catalogs.")
    names = [config.name for config in configs]
    if len(set(names)) != len(names):
        raise ValueError(f"Catalog names must be unique, got {names}.")
    if radius_arcsec <= 0:
        raise ValueError("radius_arcsec must be positive.")
    if match_flag not in (0, 1, 2):
        raise ValueError("match_flag must be 0, 1 or 2.")
    for frame, config in zip(frames, configs):
        config.validate(frame.columns)

    if len(frames[0]) == 0 or all(len(frame) == 0 for frame in frames[1:]):
        return _empty_result(names)

    #validation
    if global_maghists is not None:
        for config in configs:
            if config.name not in global_maghists:
                raise ValueError(
                    f"Missing global magnitude histograms for {config.name}."
                )

            if len(global_maghists[config.name]) != len(config.mag_columns):
                raise ValueError(
                    f"Expected {len(config.mag_columns)} histograms for "
                    f"{config.name}, but received "
                    f"{len(global_maghists[config.name])}."
                )

    area = pixel_area_deg2(aligned_order)
    # passing the precomputed global magnitude histograms to each catalog's nway table
    tables = [
        build_nway_table(
            frame,
            config,
            area,
            maghists=(
                None
                if global_maghists is None
                else global_maghists[config.name]
            ),
        )
        for frame, config in zip(frames, configs)
    ]
    #tables = [build_nway_table(frame, config, area) for frame, config in zip(frames, configs)]
    # store_mag_hists=False: otherwise nway writes histogram .pdf/.txt files to the working directory.
    results = nway_match(tables, radius_arcsec, prior_completeness, store_mag_hists=False)
    return _clean_results(results, names, match_flag, keep_unmatched)


def _clean_results(results: pd.DataFrame, names: list[str], match_flag: int, keep_unmatched: bool):
    # nway keeps the primary row number in the index (see the pandas 3 note in
    # NWAYCrossmatch._clean_nway_results); bring it back as a column.
    if names[0] in results.index.names:
        results = results.reset_index(level=names[0])
    results = results.reset_index(drop=True)

    if not keep_unmatched:
        has_counterpart = (results[names[1:]] != NO_COUNTERPART).any(axis=1)
        results = results[has_counterpart]
    if match_flag == 1:
        results = results[results["match_flag"] == 1]
    elif match_flag == 2:
        results = results[results["match_flag"] != 0]

    indices = [results[name].to_numpy(dtype=np.int64) for name in names]
    extra = results.drop(columns=names).reset_index(drop=True)
    return indices, extra


def _empty_result(names: list[str]):
    indices = [np.array([], dtype=np.int64) for _ in names]
    return indices, pd.DataFrame()
