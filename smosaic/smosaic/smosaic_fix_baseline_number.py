import math
import os

import numpy as np
import rasterio

def fix_baseline_number(input_folder: str, input_filename: str, baseline_number: str) -> str:
    """
    Adjust image value based on the baseline number in a file's naming .

    Args:
        input_folder (str): Directory path containing the input file.
        input_filename (str): Name of the input file to be processed.
        baseline_number (str): Baseline number to assign or correct in the file.

    Returns:
        str: Path or identifier of the processed file with the updated baseline number.
    """
    input_file = os.path.join(input_folder, f'{input_filename}.tif')

    with rasterio.open(input_file) as src:
        image_data = src.read()
        profile = src.profile
        height, width = src.shape

    if int(baseline_number) > 400:

        current_nodata = profile.get('nodata')
        has_nan_nodata = (
            current_nodata is not None
            and isinstance(current_nodata, float)
            and math.isnan(current_nodata)
        )

        if has_nan_nodata:
            # float32 com nodata=NaN: converte NaN → 0 antes de cast para int16
            nan_mask = np.isnan(image_data)
            new_image_data = np.where(nan_mask, np.float32(0), image_data - 1000).astype('int16')
            new_nodata = 0
        else:
            new_image_data = image_data.astype('int16') - 1000
            new_nodata = current_nodata

        profile.update({'dtype': 'int16', 'nodata': new_nodata})

        with rasterio.open(input_file, 'w', **profile) as dst:
            dst.write(new_image_data)

    return True