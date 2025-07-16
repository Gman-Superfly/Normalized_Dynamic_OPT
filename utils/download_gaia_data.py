import polars as pl
from astroquery.gaia import Gaia
import warnings
from astropy.coordinates import SkyCoord
import astropy.units as u
import os


def download_gaia_subset(limit=500, filename="gaia_data_500.csv"):
    """
    Downloads and processes a subset of the Gaia DR3 dataset using a robust,
    direct query that avoids complex table joins.

    This function queries the main gaia_source table to get parallax data,
    which is then used to calculate distances. This is a more reliable method
    than attempting to join external distance catalogs.

    The query retrieves:
    - RA, Dec, Parallax for coordinate conversion.
    - bp_rp color index, a proxy for stellar temperature.

    It then converts the spherical coordinates to 3D Cartesian coordinates
    (x, y, z) before saving.

    Args:
        limit (int): The maximum number of stars to download.
        filename (str): The name of the file to save the data to.

    Returns:
        pl.DataFrame: A polars DataFrame containing the processed data.
    """
    print("Connecting to Gaia archive... This may take a moment.")

    # This query is simplified to use parallax directly from the main source
    # table, which is more robust and avoids the previous join errors.
    adql_query = f"""
    SELECT TOP {limit}
      s.ra, s.dec, s.parallax, s.bp_rp, s.phot_g_mean_mag as mag
    FROM
      gaiadr3.gaia_source AS s
    WHERE
      s.parallax IS NOT NULL
      AND s.parallax > 0            -- Positive parallax is required for distance
      AND s.parallax_over_error > 5 -- High-quality parallax measurement
      AND s.bp_rp IS NOT NULL       -- Ensure color data exists
    ORDER BY
      s.random_index
    """

    try:
        print("Executing query to fetch stellar data...")
        job = Gaia.launch_job_async(adql_query)
        results = job.get_results()
        # Convert astropy table directly to polars (avoiding pandas dependency)
        data_dict = {col: results[col].data for col in results.colnames}
        df = pl.DataFrame(data_dict)
        print(f"Successfully downloaded data for {len(df)} stars.")

        # Calculate distance from parallax (1/parallax_in_arcsec = distance_in_parsec)
        # Parallax is in mas, so we divide by 1000 to get arcseconds.
        df = df.with_columns((1000.0 / pl.col('parallax')).alias('distance_pc'))

        print("Converting coordinates and calculating 3D positions...")
        # Use astropy to handle coordinate conversions
        coords = SkyCoord(ra=df['ra'].to_numpy()*u.deg,
                          dec=df['dec'].to_numpy()*u.deg,
                          distance=df['distance_pc'].to_numpy()*u.pc,
                          frame='icrs')

        # Get Cartesian coordinates
        df = df.with_columns([
            pl.lit(coords.cartesian.x.value).alias('x'),
            pl.lit(coords.cartesian.y.value).alias('y'),
            pl.lit(coords.cartesian.z.value).alias('z')
        ])

        # Select and save the final columns
        output_df = df.select(['x', 'y', 'z', 'bp_rp', 'mag'])

        # --- Create data directory and save file ---
        data_dir = "data"
        if not os.path.exists(data_dir):
            print(f"Creating data directory: '{data_dir}'")
            os.makedirs(data_dir)
        
        save_path = os.path.join(data_dir, filename)
        
        print(f"Saving data to {save_path}...")
        output_df.write_csv(save_path)
        print("Data successfully saved.")
        
        return output_df

    except Exception as e:
        print(f"An error occurred while querying the Gaia archive: {e}")
        print("Please check your internet connection and try again.")
        return None

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        download_gaia_subset() 