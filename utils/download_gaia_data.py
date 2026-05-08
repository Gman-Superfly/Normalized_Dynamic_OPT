from __future__ import annotations

import os
import warnings

import astropy.units as u
import polars as pl
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia


class GaiaDownloadError(RuntimeError):
    """Raised when Gaia query execution or parsing fails."""


class GaiaDataValidationError(ValueError):
    """Raised when returned Gaia data fails validation."""


def download_gaia_subset(limit: int = 500, filename: str = "gaia_data_500.csv") -> pl.DataFrame:
    """Download a Gaia DR3 subset and write Cartesian coordinates to CSV.

    Args:
        limit: Maximum number of stars requested from Gaia.
        filename: Output CSV file name under the local data directory.

    Returns:
        Processed table with x, y, z, bp_rp, and mag columns.

    Raises:
        AssertionError: If input arguments have invalid types.
        ValueError: If input values are invalid.
        GaiaDownloadError: If Gaia query execution fails.
        GaiaDataValidationError: If query output is empty or missing required columns.
    """
    assert isinstance(limit, int), f"limit must be int, got {type(limit)}"
    assert isinstance(filename, str), f"filename must be str, got {type(filename)}"
    if limit <= 0:
        raise ValueError(f"limit must be > 0, got {limit}")
    if filename.strip() == "":
        raise ValueError("filename cannot be empty")

    print("Connecting to Gaia archive, this may take a moment.")
    adql_query = f"""
    SELECT TOP {limit}
      s.ra, s.dec, s.parallax, s.bp_rp, s.phot_g_mean_mag as mag
    FROM
      gaiadr3.gaia_source AS s
    WHERE
      s.parallax IS NOT NULL
      AND s.parallax > 0
      AND s.parallax_over_error > 5
      AND s.bp_rp IS NOT NULL
    ORDER BY
      s.random_index
    """

    try:
        print("Executing query to fetch stellar data.")
        job = Gaia.launch_job_async(adql_query)
        results = job.get_results()
    except Exception as exc:
        raise GaiaDownloadError(f"Gaia query failed: {exc}") from exc

    data_dict = {col: results[col].data for col in results.colnames}
    df = pl.DataFrame(data_dict)
    if df.height == 0:
        raise GaiaDataValidationError("Gaia query returned zero rows")

    required_columns = {"ra", "dec", "parallax", "bp_rp", "mag"}
    missing_columns = required_columns.difference(set(df.columns))
    if missing_columns:
        raise GaiaDataValidationError(f"Missing required columns: {sorted(missing_columns)}")

    print(f"Successfully downloaded data for {len(df)} stars.")
    df = df.with_columns((1000.0 / pl.col("parallax")).alias("distance_pc"))

    print("Converting coordinates and calculating 3D positions.")
    coords = SkyCoord(
        ra=df["ra"].to_numpy() * u.deg,
        dec=df["dec"].to_numpy() * u.deg,
        distance=df["distance_pc"].to_numpy() * u.pc,
        frame="icrs",
    )

    df = df.with_columns(
        [
            pl.lit(coords.cartesian.x.value).alias("x"),
            pl.lit(coords.cartesian.y.value).alias("y"),
            pl.lit(coords.cartesian.z.value).alias("z"),
        ]
    )

    output_df = df.select(["x", "y", "z", "bp_rp", "mag"])
    assert output_df.height > 0, "output table cannot be empty"

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, filename)
    print(f"Saving data to {save_path}.")
    output_df.write_csv(save_path)
    print("Data successfully saved.")
    return output_df


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        download_gaia_subset()