from math import ceil

from PIL import Image
import numpy as np



def volume_to_mosaic(
        data: np.ndarray,
        axis: int = 2,
        fill_value: int = 0,
        target_ratio = 16.0 / 9.0,
        save_as = None,
        clip = None,
    ) -> np.ndarray:

    """
    Turn a 3D volume into a 2D mosaic that best matches 16:9 aspect ratio.

    Parameters
    ----------
    data : np.ndarray
        3D numpy array.
    fill_value : int, optional
        Pixel value (0-255) to use for padded (empty) tiles if n_slices does not fill grid.
    target_ratio: float
        Width-to-hight ratio of the moisaic

    Returns
    -------
    np.ndarray
        2D numpy array.
    """

    if data.ndim != 3:
        raise ValueError(f"Expected 3D image (or 4D with vol), got shape {data.shape}")
    
    data = data.swapaxes(0,1)

    # reorder so slices are along axis 0 for convenience
    if axis != 0:
        data = np.moveaxis(data, axis, 0)  # now data.shape = (nslices, H, W)

    n_slices, H, W = data.shape

    # --- choose grid (cols x rows) to best match 16:9 ---
    # each tile aspect = W/H. mosaic ratio = (cols * W) / (rows * H) = (cols/rows) * (W/H)
    # so desired cols/rows = target_ratio * (H/W)
    desired_cr = target_ratio * (H / W)
    best = None  # (error, cols, rows)
    for cols in range(1, n_slices + 1):
        rows = ceil(n_slices / cols)
        cr = cols / rows
        err = abs(cr - desired_cr)
        if best is None or err < best[0]:
            best = (err, cols, rows)
    _, cols, rows = best

    # --- create canvas for tiles (before final resize) ---
    tile_w, tile_h = W, H
    canvas_w = cols * tile_w
    canvas_h = rows * tile_h
    canvas = np.full((canvas_h, canvas_w), fill_value, dtype=data.dtype)

    # fill tiles row-major
    slice_idx = 0
    for r in range(rows):
        for c in range(cols):
            if slice_idx < n_slices:
                tile = data[slice_idx]
                y0 = r * tile_h
                x0 = c * tile_w
                canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile
            slice_idx += 1

    if clip is not None:
        canvas = np.clip(canvas, clip[0], clip[1])

    if save_as is not None:
        canvas_norm = (canvas - canvas.min()) / (canvas.max() - canvas.min())
        canvas = (canvas_norm * 255).astype(np.uint8)
        Image.fromarray(canvas).save(save_as)

    return canvas