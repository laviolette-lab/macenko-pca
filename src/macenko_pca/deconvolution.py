# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Macenko PCA stain deconvolution.

This module provides Pythonic wrappers around the high-performance Rust
implementations of the Macenko PCA stain separation method. All heavy
computation is delegated to the compiled Rust extension (``_rust``).

The input array's dtype controls which precision path is taken:

- ``float64`` → Rust f64 pipeline
- ``float32`` → Rust f32 pipeline
- ``float16`` → promoted to float32, then Rust f32 pipeline (no f16 LAPACK)
- integer types → promoted to float64 for backward compatibility

This means users can halve RAM usage simply by passing ``float32`` arrays.

Typical usage::

    import numpy as np
    from macenko_pca import rgb_separate_stains_macenko_pca, rgb_color_deconvolution

    # Compute stain matrix from an RGB image
    im_rgb = np.random.rand(256, 256, 3).astype(np.float64) * 255.0
    stain_matrix = rgb_separate_stains_macenko_pca(im_rgb)

    # Decompose the image into per-stain concentration channels
    concentrations = rgb_color_deconvolution(im_rgb, stain_matrix)

    # Reconstruct the RGB image from concentrations
    from macenko_pca import reconstruct_rgb

    im_reconstructed = reconstruct_rgb(concentrations, stain_matrix)
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from macenko_pca._rust import (
        py_color_deconvolution_f32,
        py_color_deconvolution_f64,
        py_normalize_f32,
        py_normalize_f64,
        py_reconstruct_rgb_f32,
        py_reconstruct_rgb_f64,
        py_rgb_color_deconvolution_f32,
        py_rgb_color_deconvolution_f64,
        py_rgb_separate_stains_macenko_pca_f32,
        py_rgb_separate_stains_macenko_pca_f64,
        py_rgb_to_sda_f32,
        py_rgb_to_sda_f64,
        py_separate_stains_macenko_pca_f32,
        py_separate_stains_macenko_pca_f64,
    )
except Exception as exc:
    # Provide informative stubs so importing the Python package doesn't
    # immediately fail in environments where the compiled Rust extension
    # is not present. Each stub raises a RuntimeError with actionable
    # instructions when called.
    _RUST_IMPORT_ERROR = (
        "The compiled Rust extension 'macenko_pca._rust' is not available. "
        "Build and install it to enable the high-performance implementations. "
        "For development this typically means running something like:\n\n"
        "  maturin develop\n\n"
        "or otherwise ensuring the extension is built for your Python "
        "interpreter. Original import error: "
    ) + repr(exc)

    def _missing_stub(fn_name: str):
        def _stub(*args, **kwargs):
            raise RuntimeError(
                "macenko_pca._rust::"
                + fn_name
                + " is unavailable. "
                + _RUST_IMPORT_ERROR
            )

        return _stub

    py_color_deconvolution_f32 = _missing_stub("py_color_deconvolution_f32")
    py_color_deconvolution_f64 = _missing_stub("py_color_deconvolution_f64")
    py_normalize_f32 = _missing_stub("py_normalize_f32")
    py_normalize_f64 = _missing_stub("py_normalize_f64")
    py_reconstruct_rgb_f32 = _missing_stub("py_reconstruct_rgb_f32")
    py_reconstruct_rgb_f64 = _missing_stub("py_reconstruct_rgb_f64")
    py_rgb_color_deconvolution_f32 = _missing_stub("py_rgb_color_deconvolution_f32")
    py_rgb_color_deconvolution_f64 = _missing_stub("py_rgb_color_deconvolution_f64")
    py_rgb_separate_stains_macenko_pca_f32 = _missing_stub(
        "py_rgb_separate_stains_macenko_pca_f32"
    )
    py_rgb_separate_stains_macenko_pca_f64 = _missing_stub(
        "py_rgb_separate_stains_macenko_pca_f64"
    )
    py_rgb_to_sda_f32 = _missing_stub("py_rgb_to_sda_f32")
    py_rgb_to_sda_f64 = _missing_stub("py_rgb_to_sda_f64")
    py_separate_stains_macenko_pca_f32 = _missing_stub(
        "py_separate_stains_macenko_pca_f32"
    )
    py_separate_stains_macenko_pca_f64 = _missing_stub(
        "py_separate_stains_macenko_pca_f64"
    )

# The normalize functions are expected to be provided by the compiled Rust
# extension. Importing them is handled above in the main import block. If
# the extension is not present, the names above will be the missing-stub
# callables created in the except block above, which raise informative
# RuntimeErrors when invoked. No Python fallback is supplied here; rebuild
# the Rust extension to provide these symbols.

# Type alias for the supported return types.
_FloatArray = Union[NDArray[np.float32], NDArray[np.float64]]

# ---------------------------------------------------------------------------
# Well-known stain colour vectors
# ---------------------------------------------------------------------------

#: Mapping of common histology stain names to their reference colour vectors
#: in SDA (stain-density-absorbance) space.  Each value is a 3-element list
#: representing the unit-direction of that stain.  The ``'null'`` entry is
#: a zero vector used as a placeholder when no stain is present.
#:
#: These vectors originate from the HistomicsTK project and are widely used
#: as reference directions when identifying which column of an adaptively
#: estimated stain matrix corresponds to a particular biological stain.
stain_color_map: dict[str, list[float]] = {
    "hematoxylin": [0.65, 0.70, 0.29],
    "eosin": [0.07, 0.99, 0.11],
    "dab": [0.27, 0.57, 0.78],
    "null": [0.0, 0.0, 0.0],
}


def normalize(a: ArrayLike) -> NDArray[np.float64]:
    """Normalize an array to unit norm using the Rust backend.

    For a 1-D array the vector is divided by its L2 norm. For a 2-D array
    each column is independently normalised. Zero-norm vectors / columns
    are left as zeros.

    This function assumes the compiled Rust extension exposes
    `py_normalize_f64` (registered above). If the extension is missing,
    the previously-installed import-time stub will raise an informative
    RuntimeError explaining how to rebuild/install the extension.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        # reshape to (N, 1) and call the Rust normalizer which expects 2-D input
        col = np.ascontiguousarray(a.reshape(-1, 1))
        result = np.asarray(py_normalize_f64(col))
        return result.ravel()
    # 2-D: delegate directly to Rust
    return np.asarray(py_normalize_f64(np.ascontiguousarray(a)))


def find_stain_index(
    reference: ArrayLike,
    w: ArrayLike,
) -> int:
    """Identify the column of *w* that best aligns with *reference*.

    This is used with adaptive deconvolution routines where the order of
    returned stain vectors is not guaranteed.  The function identifies the
    stain vector (column) of *w* whose direction most closely matches the
    provided *reference* direction.

    Vectors are normalised to unit length before comparison so alignment
    is measured purely by angle (cosine similarity), not magnitude.

    :param reference: 1-D array (length 3) representing the query stain
        vector.  Typically one of the values from :data:`stain_color_map`.
    :type reference: ArrayLike
    :param w: A ``(3, N)`` array whose columns are stain vectors to search.
        Usually a ``(3, 3)`` stain matrix returned by
        :func:`rgb_separate_stains_macenko_pca`.
    :type w: ArrayLike
    :return: Column index of the stain vector in *w* with the best
        alignment to *reference*.
    :rtype: int

    Notes
    -----
    Alignment is determined by the absolute value of the dot product between
    unit vectors so that anti-parallel vectors (pointing in exactly opposite
    directions) are treated as equivalent.

    Examples
    --------
    >>> import numpy as np
    >>> from macenko_pca import find_stain_index, stain_color_map
    >>> w = np.eye(3)
    >>> find_stain_index(stain_color_map["hematoxylin"], w)
    1

    See Also
    --------
    stain_color_map : Well-known reference stain vectors.
    rgb_separate_stains_macenko_pca : Estimate stain matrix from an RGB image.
    """
    reference = np.asarray(reference, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    dot_products = np.dot(normalize(reference), normalize(w))
    return int(np.argmax(np.abs(dot_products)))


def _resolve_dtype(arr: np.ndarray) -> np.ndarray:
    """Coerce *arr* to a Rust-compatible float dtype, preserving precision.

    - float64 → kept as-is
    - float32 → kept as-is
    - float16 → promoted to float32 (no f16 LAPACK)
    - integer / other → promoted to float64 (backward compat)

    :param arr: input array (any dtype)
    :type arr: numpy.ndarray
    :return: array guaranteed to be float32 or float64
    :rtype: numpy.ndarray
    """
    if arr.dtype == np.float64:
        return arr
    if arr.dtype == np.float32:
        return arr
    if arr.dtype == np.float16:
        return arr.astype(np.float32)
    # integers, bools, other → float64 for backward compat
    return arr.astype(np.float64)


def _is_f32(arr: np.ndarray) -> bool:
    """Return ``True`` when *arr* should use the f32 Rust path."""
    return arr.dtype == np.float32


# ---------------------------------------------------------------------------
# Stain matrix estimation
# ---------------------------------------------------------------------------


def rgb_separate_stains_macenko_pca(
    im_rgb: ArrayLike,
    bg_int: list[float] | None = None,
    minimum_magnitude: float = 16.0,
    min_angle_percentile: float = 0.01,
    max_angle_percentile: float = 0.99,
    mask_out: ArrayLike | None = None,
) -> _FloatArray:
    """Compute the stain matrix from an RGB image using the Macenko PCA method.

    Converts the RGB image to SDA (stain-density-absorbance) space and then
    applies PCA-based angle binning to identify the two principal stain
    vectors.  A third complementary stain vector is generated via cross
    product.

    The **dtype of *im_rgb*** controls the computation precision:

    - ``float64`` → full f64 pipeline
    - ``float32`` → f32 pipeline (≈ half the RAM)
    - ``float16`` → promoted to f32, then f32 pipeline
    - integer types → promoted to f64

    :param im_rgb: Input RGB image with shape ``(H, W, 3)``.  Values are
        expected in the range ``[0, 255]``.
    :type im_rgb: ArrayLike
    :param bg_int: Background intensity per channel, used for the SDA
        transform.  A single-element list broadcasts to all channels.
        Defaults to ``[256.0, 256.0, 256.0]``.
    :type bg_int: list[float] | None
    :param minimum_magnitude: Minimum magnitude threshold for filtering
        projected pixels in PCA space.
    :type minimum_magnitude: float
    :param min_angle_percentile: Lower angle percentile for selecting the
        first stain vector (0-1 range).
    :type min_angle_percentile: float
    :param max_angle_percentile: Upper angle percentile for selecting the
        second stain vector (0-1 range).
    :type max_angle_percentile: float
    :param mask_out: Optional boolean mask with shape ``(H, W)``.  Pixels
        where the mask is ``True`` are excluded from the computation.
    :type mask_out: ArrayLike | None
    :return: A ``(3, 3)`` stain matrix where each column is a normalised
        stain vector.  The third column is the cross-product complement of
        the first two.  Dtype matches the computation precision.
    :rtype: NDArray[np.float32] | NDArray[np.float64]
    :raises ValueError: If the input shapes are incompatible or the SDA
        conversion fails.
    """
    im_rgb = _resolve_dtype(np.asarray(im_rgb))
    if im_rgb.ndim != 3 or im_rgb.shape[2] != 3:
        msg = f"im_rgb must have shape (H, W, 3), got {im_rgb.shape}"
        raise ValueError(msg)

    mask = None
    if mask_out is not None:
        mask = np.asarray(mask_out, dtype=bool)
        if mask.shape != im_rgb.shape[:2]:
            msg = (
                f"mask_out shape {mask.shape} does not match "
                f"image spatial dimensions {im_rgb.shape[:2]}"
            )
            raise ValueError(msg)

    fn = (
        py_rgb_separate_stains_macenko_pca_f32
        if _is_f32(im_rgb)
        else py_rgb_separate_stains_macenko_pca_f64
    )

    if bg_int is None:
        _bg_int = None
    elif isinstance(bg_int, (int, float, np.integer, np.floating)):
        _bg_int = [float(bg_int)]
    else:
        _bg_int = list(bg_int)

    return np.asarray(
        fn(
            im_rgb,
            bg_int=bg_int,
            minimum_magnitude=minimum_magnitude,
            min_angle_percentile=min_angle_percentile,
            max_angle_percentile=max_angle_percentile,
            mask_out=mask,
        )
    )


def separate_stains_macenko_pca(
    im_sda: ArrayLike,
    minimum_magnitude: float = 16.0,
    min_angle_percentile: float = 0.01,
    max_angle_percentile: float = 0.99,
    mask_out: ArrayLike | None = None,
) -> _FloatArray:
    """Compute the stain matrix from an SDA image using the Macenko PCA method.

    This is the lower-level entry point that operates directly on images
    already in SDA (stain-density-absorbance) space. Use
    :func:`rgb_separate_stains_macenko_pca` if you are starting from an RGB
    image.

    Precision is controlled by the dtype of *im_sda* (see
    :func:`rgb_separate_stains_macenko_pca` for details).

    :param im_sda: Input image in SDA space with shape ``(H, W, C)``.
    :type im_sda: ArrayLike
    :param minimum_magnitude: Minimum magnitude threshold for filtering
        projected pixels in PCA space.
    :type minimum_magnitude: float
    :param min_angle_percentile: Lower angle percentile (0-1 range).
    :type min_angle_percentile: float
    :param max_angle_percentile: Upper angle percentile (0-1 range).
    :type max_angle_percentile: float
    :param mask_out: Optional boolean mask with shape ``(H, W)``.  Pixels
        where the mask is ``True`` are excluded.
    :type mask_out: ArrayLike | None
    :return: A ``(3, 3)`` stain matrix.
    :rtype: NDArray[np.float32] | NDArray[np.float64]
    :raises ValueError: If the input is not 3-dimensional.
    """
    im_sda = _resolve_dtype(np.asarray(im_sda))
    if im_sda.ndim != 3:
        msg = f"im_sda must be 3D, got {im_sda.ndim}D"
        raise ValueError(msg)

    mask = None
    if mask_out is not None:
        mask = np.asarray(mask_out, dtype=bool)
        if mask.shape != im_sda.shape[:2]:
            msg = (
                f"mask_out shape {mask.shape} does not match "
                f"image spatial dimensions {im_sda.shape[:2]}"
            )
            raise ValueError(msg)

    fn = (
        py_separate_stains_macenko_pca_f32
        if _is_f32(im_sda)
        else py_separate_stains_macenko_pca_f64
    )

    return np.asarray(
        fn(
            im_sda,
            minimum_magnitude=minimum_magnitude,
            min_angle_percentile=min_angle_percentile,
            max_angle_percentile=max_angle_percentile,
            mask_out=mask,
        )
    )


# ---------------------------------------------------------------------------
# RGB ↔ SDA conversion
# ---------------------------------------------------------------------------


def rgb_to_sda(
    im_rgb: ArrayLike,
    bg_int: list[float] | None = None,
    allow_negative: bool = False,
) -> _FloatArray:
    """Convert an RGB image or matrix to SDA (stain-density-absorbance) space.

    Precision is controlled by the dtype of *im_rgb* (see
    :func:`rgb_separate_stains_macenko_pca` for the dtype mapping rules).

    :param im_rgb: Input RGB data.  May be a 3D image of shape ``(H, W, C)``
        or a 2D matrix of shape ``(N, C)``.
    :type im_rgb: ArrayLike
    :param bg_int: Background intensity per channel.  A single-element list
        broadcasts to all channels.  Defaults to ``[256.0, ...]``.
    :type bg_int: list[float] | None
    :param allow_negative: If ``False`` (default), negative SDA values are
        clamped to zero.
    :type allow_negative: bool
    :return: The image or matrix in SDA space, same shape as the input.
        Dtype matches the computation precision.
    :rtype: NDArray[np.float32] | NDArray[np.float64]
    :raises ValueError: If the input dimensionality is unsupported.
    """
    im_rgb = _resolve_dtype(np.asarray(im_rgb))
    if im_rgb.ndim not in (2, 3):
        msg = f"im_rgb must be 2D or 3D, got {im_rgb.ndim}D"
        raise ValueError(msg)

    fn = py_rgb_to_sda_f32 if _is_f32(im_rgb) else py_rgb_to_sda_f64

    return np.asarray(
        fn(
            im_rgb,
            bg_int=bg_int,
            allow_negative=allow_negative,
        )
    )


# ---------------------------------------------------------------------------
# Colour deconvolution (applying stain vectors)
# ---------------------------------------------------------------------------


def color_deconvolution(
    im_sda: ArrayLike,
    stain_matrix: ArrayLike,
) -> _FloatArray:
    """Decompose an SDA image into per-stain concentration channels.

    Given an image in SDA (stain-density-absorbance) space and a stain
    matrix **W** (3x3, columns = stain vectors), this function computes
    the per-pixel stain concentrations by solving:

        OD = W x c => c = W⁻¹ x OD

    In matrix form for all pixels simultaneously:

        concentrations = SDA_pixels x (W⁻¹)ᵀ

    The returned array has the same spatial dimensions as the input; each
    channel *i* holds the concentration of stain *i*.

    Use :func:`rgb_color_deconvolution` if you are starting from an RGB
    image and want the conversion handled automatically.

    Precision is controlled by the dtype of *im_sda* (see
    :func:`rgb_separate_stains_macenko_pca` for the dtype mapping rules).

    :param im_sda: Input image in SDA space with shape ``(H, W, 3)``.
    :type im_sda: ArrayLike
    :param stain_matrix: A ``(3, 3)`` stain matrix whose columns are
        normalised stain vectors (as returned by
        :func:`rgb_separate_stains_macenko_pca`).
    :type stain_matrix: ArrayLike
    :return: A ``(H, W, 3)`` concentration image.  Channel *i* is the
        concentration of stain *i*.  Dtype matches the computation
        precision.
    :rtype: NDArray[np.float32] | NDArray[np.float64]
    :raises ValueError: If shapes are incompatible or the stain matrix is
        singular.

    Example::

        from macenko_pca import (
            rgb_to_sda,
            rgb_separate_stains_macenko_pca,
            color_deconvolution,
        )

        im_sda = rgb_to_sda(im_rgb)
        stain_matrix = rgb_separate_stains_macenko_pca(im_rgb)
        concentrations = color_deconvolution(im_sda, stain_matrix)

        # Channel 0 = first stain, channel 1 = second stain, channel 2 = residual
        hematoxylin = concentrations[:, :, 0]
        eosin = concentrations[:, :, 1]
    """
    im_sda = _resolve_dtype(np.asarray(im_sda))
    stain_matrix = np.asarray(stain_matrix, dtype=im_sda.dtype)

    if im_sda.ndim != 3 or im_sda.shape[2] != 3:
        msg = f"im_sda must have shape (H, W, 3), got {im_sda.shape}"
        raise ValueError(msg)
    if stain_matrix.shape != (3, 3):
        msg = f"stain_matrix must be (3, 3), got {stain_matrix.shape}"
        raise ValueError(msg)

    fn = py_color_deconvolution_f32 if _is_f32(im_sda) else py_color_deconvolution_f64

    return np.asarray(fn(im_sda, stain_matrix))


def rgb_color_deconvolution(
    im_rgb: ArrayLike,
    stain_matrix: ArrayLike,
    bg_int: list[float] | None = None,
) -> _FloatArray:
    """Decompose an RGB image into per-stain concentration channels.

    Convenience wrapper that converts the RGB image to SDA space and then
    applies :func:`color_deconvolution` in a single call.

    Precision is controlled by the dtype of *im_rgb* (see
    :func:`rgb_separate_stains_macenko_pca` for the dtype mapping rules).

    :param im_rgb: Input RGB image with shape ``(H, W, 3)``.  Values are
        expected in the range ``[0, 255]``.
    :type im_rgb: ArrayLike
    :param stain_matrix: A ``(3, 3)`` stain matrix whose columns are
        normalised stain vectors.
    :type stain_matrix: ArrayLike
    :param bg_int: Background intensity per channel for the SDA transform.
        A single-element list broadcasts to all channels.  Defaults to
        ``[256.0, 256.0, 256.0]``.
    :type bg_int: list[float] | None
    :return: A ``(H, W, 3)`` concentration image.
    :rtype: NDArray[np.float32] | NDArray[np.float64]
    :raises ValueError: If shapes are incompatible or the stain matrix is
        singular.

    Example::

        import numpy as np
        from macenko_pca import rgb_separate_stains_macenko_pca, rgb_color_deconvolution

        im_rgb = np.random.rand(256, 256, 3) * 255.0
        stain_matrix = rgb_separate_stains_macenko_pca(im_rgb)
        concentrations = rgb_color_deconvolution(im_rgb, stain_matrix)

        # Visualise the hematoxylin channel
        hematoxylin = concentrations[:, :, 0]
    """
    im_rgb = _resolve_dtype(np.asarray(im_rgb))
    stain_matrix = np.asarray(stain_matrix, dtype=im_rgb.dtype)

    if im_rgb.ndim != 3 or im_rgb.shape[2] != 3:
        msg = f"im_rgb must have shape (H, W, 3), got {im_rgb.shape}"
        raise ValueError(msg)
    if stain_matrix.shape != (3, 3):
        msg = f"stain_matrix must be (3, 3), got {stain_matrix.shape}"
        raise ValueError(msg)

    fn = (
        py_rgb_color_deconvolution_f32
        if _is_f32(im_rgb)
        else py_rgb_color_deconvolution_f64
    )

    return np.asarray(fn(im_rgb, stain_matrix, bg_int=bg_int))


def reconstruct_rgb(
    concentrations: ArrayLike,
    stain_matrix: ArrayLike,
    bg_int: float | None = None,
) -> _FloatArray:
    """Reconstruct an RGB image from stain concentrations and a stain matrix.

    Inverts the deconvolution process performed by
    :func:`color_deconvolution` / :func:`rgb_color_deconvolution`:

    1. Recompute the SDA image: ``SDA = concentrations x Wᵀ``
    2. Convert SDA back to RGB:
       ``RGB_ch = bg x exp(-SDA_ch x ln(bg) / 255)``

    This is useful for stain normalisation workflows where you modify the
    concentration channels (e.g. scale, zero-out a stain) and then
    reconstruct the image.

    Precision is controlled by the dtype of *concentrations* (see
    :func:`rgb_separate_stains_macenko_pca` for the dtype mapping rules).

    :param concentrations: Stain concentration image with shape
        ``(H, W, 3)`` (as returned by :func:`color_deconvolution`).
    :type concentrations: ArrayLike
    :param stain_matrix: The ``(3, 3)`` stain matrix used during
        deconvolution.
    :type stain_matrix: ArrayLike
    :param bg_int: Background intensity used during the original SDA
        transform.  Applied uniformly to all channels.  Defaults to
        ``256.0``.
    :type bg_int: float | None
    :return: A ``(H, W, 3)`` reconstructed RGB image with values clamped
        to ``[0, bg_int]``.
    :rtype: NDArray[np.float32] | NDArray[np.float64]
    :raises ValueError: If shapes are incompatible.

    Example::

        from macenko_pca import (
            rgb_separate_stains_macenko_pca,
            rgb_color_deconvolution,
            reconstruct_rgb,
        )

        stain_matrix = rgb_separate_stains_macenko_pca(im_rgb)
        concentrations = rgb_color_deconvolution(im_rgb, stain_matrix)

        # Zero-out the eosin channel and reconstruct
        concentrations[:, :, 1] = 0.0
        hematoxylin_only = reconstruct_rgb(concentrations, stain_matrix)
    """
    concentrations = _resolve_dtype(np.asarray(concentrations))
    stain_matrix = np.asarray(stain_matrix, dtype=concentrations.dtype)

    if concentrations.ndim != 3 or concentrations.shape[2] != 3:
        msg = f"concentrations must have shape (H, W, 3), got {concentrations.shape}"
        raise ValueError(msg)
    if stain_matrix.shape != (3, 3):
        msg = f"stain_matrix must be (3, 3), got {stain_matrix.shape}"
        raise ValueError(msg)

    fn = py_reconstruct_rgb_f32 if _is_f32(concentrations) else py_reconstruct_rgb_f64

    return np.asarray(fn(concentrations, stain_matrix, bg_int=bg_int))
