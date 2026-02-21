# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Cross-validation tests against HistomicsTK reference implementations.

Every public function in the deconvolution pipeline has a pure-NumPy /
SciPy reference implementation inlined here that mirrors the HistomicsTK
logic.  Tests compare our Rust-backed outputs against those references and
assert that they agree to within a stated tolerance.

The Rust code can produce *very slightly* different results due to
differences in LAPACK implementations, floating-point operation ordering,
and parallelism.  Tolerances are chosen to be tight enough to catch real
bugs but loose enough to accommodate these numerical differences.

Test classes
------------
TestRgbToSdaParity
    SDA conversion parity.
TestNormalizeParity
    Column-wise normalize parity (Rust vs NumPy).
TestColorDeconvolutionParity
    Given the *same* stain matrix, compare concentration outputs.
TestReconstructRgbParity
    Given the *same* concentrations and stain matrix, compare RGB outputs.
TestSeparateStainsMacenkoPcaParity
    Full stain-matrix estimation parity (angle-aware comparison).
TestEndToEndPipelineParity
    Full pipeline: RGB -> stain matrix -> concentrations -> reconstruction.
"""

from __future__ import annotations

import numpy as np
import pytest

from macenko_pca.deconvolution import (
    color_deconvolution,
    find_stain_index,
    normalize,
    reconstruct_rgb,
    rgb_color_deconvolution,
    rgb_separate_stains_macenko_pca,
    rgb_to_sda,
    separate_stains_macenko_pca,
    stain_color_map,
)

# =====================================================================
# HistomicsTK reference implementations (pure NumPy)
# =====================================================================


def _htk_normalize(a: np.ndarray) -> np.ndarray:
    """HistomicsTK linalg.normalize — unit-norm columns (2-D) or vector (1-D)."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        n = np.linalg.norm(a)
        return a / n if n != 0.0 else np.zeros_like(a)
    norms = np.linalg.norm(a, axis=0, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return a / norms


def _htk_rgb_to_sda(
    im_rgb: np.ndarray,
    bg_int: list[float] | None = None,
    allow_negative: bool = False,
) -> np.ndarray:
    """Reference rgb_to_sda matching HistomicsTK's colour_conversion module.

    SDA_c = -log(I_c / I_0) * (255 / log(I_0))

    where I_0 is the per-channel background intensity (default 256).
    """
    im = np.asarray(im_rgb, dtype=np.float64)
    if bg_int is None:
        bg = np.array([256.0, 256.0, 256.0])
    else:
        bg = np.array(bg_int, dtype=np.float64)
        if bg.size == 1:
            bg = np.full(3, bg.item())

    eps = 1e-10
    im_safe = np.maximum(im, eps)
    sda = -np.log(im_safe / bg) * (255.0 / np.log(bg))
    if not allow_negative:
        sda = np.maximum(sda, 0.0)
    return sda


def _htk_color_deconvolution(
    im_sda: np.ndarray,
    stain_matrix: np.ndarray,
) -> np.ndarray:
    """Reference color_deconvolution: C = SDA @ inv(W)^T."""
    h, w, _c = im_sda.shape
    sda_flat = im_sda.reshape(-1, 3)
    w_inv = np.linalg.inv(stain_matrix)
    result = sda_flat @ w_inv.T
    return result.reshape(h, w, 3)


def _htk_reconstruct_rgb(
    concentrations: np.ndarray,
    stain_matrix: np.ndarray,
    bg_int: float = 256.0,
) -> np.ndarray:
    """Reference reconstruct_rgb: invert the SDA -> RGB transform."""
    h, w, _c = concentrations.shape
    conc_flat = concentrations.reshape(-1, 3)
    sda = conc_flat @ stain_matrix.T
    ln_bg = np.log(bg_int)
    scale = ln_bg / 255.0
    rgb = bg_int * np.exp(-sda * scale)
    rgb = np.clip(rgb, 0.0, bg_int)
    return rgb.reshape(h, w, 3)


def _htk_complement_stain_matrix(w: np.ndarray) -> np.ndarray:
    """Reference complement: replace column 2 with normalised cross-product."""
    s0 = w[:, 0]
    s1 = w[:, 1]
    s2 = np.cross(s0, s1)
    n = np.linalg.norm(s2)
    if n > 0:
        s2 = s2 / n
    out = np.zeros((3, 3), dtype=np.float64)
    out[:, 0] = s0
    out[:, 1] = s1
    out[:, 2] = s2
    return out


def _htk_separate_stains_macenko_pca(
    im_sda: np.ndarray,
    minimum_magnitude: float = 16.0,
    min_angle_percentile: float = 0.01,
    max_angle_percentile: float = 0.99,
    mask_out: np.ndarray | None = None,
) -> np.ndarray:
    """Reference Macenko PCA stain separation (pure NumPy + scipy SVD).

    This closely follows the HistomicsTK implementation in
    ``preprocessing.color_deconvolution.separate_stains_macenko_pca``.
    """
    _h, _w, c = im_sda.shape
    # Flatten to (3, N)
    m = im_sda.reshape(-1, c).T.astype(np.float64)

    # Apply mask
    if mask_out is not None:
        keep = ~mask_out.ravel()
        m = m[:, keep]

    # Remove non-finite columns
    finite_mask = np.all(np.isfinite(m), axis=0)
    m = m[:, finite_mask]

    # SVD for principal components
    u, _, _ = np.linalg.svd(m, full_matrices=True)

    # Project into PCA plane (first 2 components)
    proj = u[:, :2].T @ m  # (2, N)

    # Magnitude filter
    mag = np.linalg.norm(proj, axis=0)
    filt = proj[:, mag > minimum_magnitude]

    # Angles
    filt_norm = np.linalg.norm(filt, axis=0, keepdims=True)
    filt_norm = np.where(filt_norm == 0, 1.0, filt_norm)
    fn = filt / filt_norm
    angles = (1.0 - fn[1, :]) * np.sign(fn[0, :])

    def _get_percentile_vector(pcs, filt_data, angs, p):
        size = len(angs)
        idx = int(min(p * size + 0.5, size - 1))
        order = np.argpartition(angs, idx)
        chosen = order[idx]
        return pcs[:, :2] @ filt_data[:, chosen]

    min_v = _get_percentile_vector(u, filt, angles, min_angle_percentile)
    max_v = _get_percentile_vector(u, filt, angles, max_angle_percentile)

    stains = np.column_stack([min_v, max_v])
    stains = _htk_normalize(stains)
    return _htk_complement_stain_matrix(stains)


def _cosine_similarity_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute cosine similarity between corresponding columns of a and b."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sims = []
    for i in range(a.shape[1]):
        na = np.linalg.norm(a[:, i])
        nb = np.linalg.norm(b[:, i])
        if na > 0 and nb > 0:
            sims.append(abs(np.dot(a[:, i], b[:, i]) / (na * nb)))
        else:
            sims.append(0.0)
    return np.array(sims)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_rgb(rng):
    """64x64 synthetic RGB image, float64, values in (1, 255)."""
    return rng.uniform(1.0, 255.0, size=(64, 64, 3))


@pytest.fixture
def sample_rgb_f32(sample_rgb):
    return sample_rgb.astype(np.float32)


@pytest.fixture
def he_like_image(rng):
    """128x128 image synthesised from H&E-like stain directions.

    This gives the PCA estimator something meaningful to latch onto so we
    can compare estimated stain matrices between implementations.
    """
    h = np.array([0.65, 0.70, 0.29])
    e = np.array([0.07, 0.99, 0.11])
    rows, cols = 128, 128
    n = rows * cols
    c_h = rng.uniform(0.1, 1.0, n)
    c_e = rng.uniform(0.1, 1.0, n)
    od = np.outer(c_h, h) + np.outer(c_e, e)
    od += rng.normal(0, 0.02, od.shape)
    od = np.clip(od, 0, None)
    rgb = 256.0 * np.exp(-od)
    return np.clip(rgb, 1.0, 255.0).reshape(rows, cols, 3)


@pytest.fixture
def known_stain_matrix():
    """A hand-built normalised H&E stain matrix with cross-product complement."""
    h = np.array([0.65, 0.70, 0.29])
    e = np.array([0.07, 0.99, 0.11])
    h = h / np.linalg.norm(h)
    e = e / np.linalg.norm(e)
    c = np.cross(h, e)
    c = c / np.linalg.norm(c)
    return np.column_stack([h, e, c])


# =====================================================================
# rgb_to_sda parity
# =====================================================================


class TestRgbToSdaParity:
    """Compare our Rust-backed rgb_to_sda against the NumPy reference."""

    def test_default_bg_f64(self, sample_rgb):
        ours = rgb_to_sda(sample_rgb)
        ref = _htk_rgb_to_sda(sample_rgb)
        np.testing.assert_allclose(ours, ref, atol=1e-10, rtol=1e-10)

    def test_default_bg_f32(self, sample_rgb_f32):
        ours = rgb_to_sda(sample_rgb_f32)
        ref = _htk_rgb_to_sda(sample_rgb_f32)
        np.testing.assert_allclose(ours, ref, atol=1e-4, rtol=1e-4)

    def test_custom_bg_per_channel(self, sample_rgb):
        bg = [240.0, 250.0, 245.0]
        ours = rgb_to_sda(sample_rgb, bg_int=bg)
        ref = _htk_rgb_to_sda(sample_rgb, bg_int=bg)
        np.testing.assert_allclose(ours, ref, atol=1e-10, rtol=1e-10)

    def test_allow_negative(self, sample_rgb):
        ours = rgb_to_sda(sample_rgb, allow_negative=True)
        ref = _htk_rgb_to_sda(sample_rgb, allow_negative=True)
        np.testing.assert_allclose(ours, ref, atol=1e-10, rtol=1e-10)

    def test_non_negative_clamp_matches(self, sample_rgb):
        """Both should clamp negative SDA values to 0 by default."""
        ours = rgb_to_sda(sample_rgb)
        ref = _htk_rgb_to_sda(sample_rgb)
        assert np.all(ours >= 0.0)
        assert np.all(ref >= 0.0)
        np.testing.assert_allclose(ours, ref, atol=1e-10, rtol=1e-10)

    def test_he_like_image(self, he_like_image):
        ours = rgb_to_sda(he_like_image)
        ref = _htk_rgb_to_sda(he_like_image)
        np.testing.assert_allclose(ours, ref, atol=1e-10, rtol=1e-10)


# =====================================================================
# normalize parity
# =====================================================================


class TestNormalizeParity:
    """Compare our Rust-backed normalize against the NumPy reference."""

    def test_1d_vector(self):
        v = np.array([3.0, 4.0, 0.0])
        ours = normalize(v)
        ref = _htk_normalize(v)
        np.testing.assert_allclose(ours, ref, atol=1e-14)

    def test_1d_unit_vector(self):
        v = np.array([0.0, 1.0, 0.0])
        ours = normalize(v)
        ref = _htk_normalize(v)
        np.testing.assert_allclose(ours, ref, atol=1e-14)

    def test_1d_zero_vector(self):
        v = np.array([0.0, 0.0, 0.0])
        ours = normalize(v)
        ref = _htk_normalize(v)
        np.testing.assert_array_equal(ours, ref)

    def test_2d_identity(self):
        m = np.eye(3)
        ours = normalize(m)
        ref = _htk_normalize(m)
        np.testing.assert_allclose(ours, ref, atol=1e-14)

    def test_2d_random(self, rng):
        m = rng.random((3, 5))
        ours = normalize(m)
        ref = _htk_normalize(m)
        np.testing.assert_allclose(ours, ref, atol=1e-14)

    def test_2d_with_zero_column(self):
        m = np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        ours = normalize(m)
        ref = _htk_normalize(m)
        np.testing.assert_allclose(ours, ref, atol=1e-14)

    def test_columns_have_unit_norm(self, rng):
        m = rng.random((3, 4)) * 100.0
        result = normalize(m)
        norms = np.linalg.norm(result, axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-14)

    def test_stain_color_map_vectors(self):
        """Normalizing stain_color_map vectors matches reference."""
        for name, vec in stain_color_map.items():
            if name == "null":
                continue
            ours = normalize(np.array(vec))
            ref = _htk_normalize(np.array(vec))
            np.testing.assert_allclose(
                ours, ref, atol=1e-14, err_msg=f"Mismatch for {name}"
            )

    def test_stain_matrix(self, known_stain_matrix):
        ours = normalize(known_stain_matrix)
        ref = _htk_normalize(known_stain_matrix)
        np.testing.assert_allclose(ours, ref, atol=1e-14)

    def test_large_random_matrix(self, rng):
        m = rng.random((3, 1000))
        ours = normalize(m)
        ref = _htk_normalize(m)
        np.testing.assert_allclose(ours, ref, atol=1e-12)


# =====================================================================
# color_deconvolution parity
# =====================================================================


class TestColorDeconvolutionParity:
    """Given the *same* stain matrix, compare concentration outputs."""

    def test_known_stain_matrix_f64(self, sample_rgb, known_stain_matrix):
        sda = rgb_to_sda(sample_rgb)
        ours = color_deconvolution(sda, known_stain_matrix)
        ref = _htk_color_deconvolution(sda.astype(np.float64), known_stain_matrix)
        np.testing.assert_allclose(ours, ref, atol=1e-8, rtol=1e-8)

    def test_known_stain_matrix_f32(self, sample_rgb_f32, known_stain_matrix):
        sda = rgb_to_sda(sample_rgb_f32)
        w32 = known_stain_matrix.astype(np.float32)
        ours = color_deconvolution(sda, w32)
        ref = _htk_color_deconvolution(
            sda.astype(np.float64),
            known_stain_matrix,
        )
        np.testing.assert_allclose(ours, ref, atol=1e-2, rtol=1e-2)

    def test_identity_stain_matrix(self, sample_rgb):
        """Identity stain matrix: concentrations should equal SDA values."""
        sda = rgb_to_sda(sample_rgb)
        ours = color_deconvolution(sda, np.eye(3))
        ref = _htk_color_deconvolution(sda.astype(np.float64), np.eye(3))
        np.testing.assert_allclose(ours, ref, atol=1e-10, rtol=1e-10)

    def test_he_like_image(self, he_like_image, known_stain_matrix):
        sda = rgb_to_sda(he_like_image)
        ours = color_deconvolution(sda, known_stain_matrix)
        ref = _htk_color_deconvolution(sda.astype(np.float64), known_stain_matrix)
        np.testing.assert_allclose(ours, ref, atol=1e-8, rtol=1e-8)

    def test_rgb_color_deconvolution_convenience(self, sample_rgb, known_stain_matrix):
        """The RGB convenience wrapper should match manual SDA + deconvolve."""
        ours = rgb_color_deconvolution(sample_rgb, known_stain_matrix)
        sda = _htk_rgb_to_sda(sample_rgb)
        ref = _htk_color_deconvolution(sda, known_stain_matrix)
        np.testing.assert_allclose(ours, ref, atol=1e-8, rtol=1e-8)

    def test_all_finite(self, sample_rgb, known_stain_matrix):
        sda = rgb_to_sda(sample_rgb)
        ours = color_deconvolution(sda, known_stain_matrix)
        assert np.all(np.isfinite(ours))


# =====================================================================
# reconstruct_rgb parity
# =====================================================================


class TestReconstructRgbParity:
    """Given the same concentrations and stain matrix, compare RGB outputs."""

    def test_default_bg_f64(self, sample_rgb, known_stain_matrix):
        sda = rgb_to_sda(sample_rgb)
        conc = _htk_color_deconvolution(sda.astype(np.float64), known_stain_matrix)
        ours = reconstruct_rgb(conc, known_stain_matrix)
        ref = _htk_reconstruct_rgb(conc, known_stain_matrix)
        np.testing.assert_allclose(ours, ref, atol=1e-8, rtol=1e-8)

    def test_custom_bg(self, sample_rgb, known_stain_matrix):
        bg = 240.0
        sda = rgb_to_sda(sample_rgb, bg_int=[bg])
        conc = _htk_color_deconvolution(sda.astype(np.float64), known_stain_matrix)
        ours = reconstruct_rgb(conc, known_stain_matrix, bg_int=bg)
        ref = _htk_reconstruct_rgb(conc, known_stain_matrix, bg_int=bg)
        np.testing.assert_allclose(ours, ref, atol=1e-8, rtol=1e-8)

    def test_clamped_to_valid_range(self, sample_rgb, known_stain_matrix):
        sda = rgb_to_sda(sample_rgb)
        conc = color_deconvolution(sda, known_stain_matrix)
        recon = reconstruct_rgb(conc, known_stain_matrix)
        assert np.all(recon >= 0.0)
        assert np.all(recon <= 256.0)

    def test_roundtrip_parity_f64(self, sample_rgb, known_stain_matrix):
        """Full roundtrip: RGB -> SDA -> conc -> reconstruct ~ original."""
        conc = rgb_color_deconvolution(sample_rgb, known_stain_matrix)
        ours = reconstruct_rgb(conc, known_stain_matrix)
        ref = _htk_reconstruct_rgb(
            conc.astype(np.float64),
            known_stain_matrix,
        )
        np.testing.assert_allclose(ours, ref, atol=1e-8, rtol=1e-8)

    def test_zero_concentrations(self, known_stain_matrix):
        """Zero concentrations should reconstruct to the background level."""
        conc = np.zeros((10, 10, 3), dtype=np.float64)
        ours = reconstruct_rgb(conc, known_stain_matrix)
        ref = _htk_reconstruct_rgb(conc, known_stain_matrix)
        np.testing.assert_allclose(ours, ref, atol=1e-10)


# =====================================================================
# separate_stains_macenko_pca parity
# =====================================================================


class TestSeparateStainsMacenkoPcaParity:
    """Compare stain matrix estimation between Rust and reference.

    SVD sign ambiguity and floating-point ordering differences mean the
    columns may be flipped or negated.  We compare using cosine similarity
    after using find_stain_index to align columns.
    """

    # Tolerance for cosine similarity between matched stain vectors.
    # The Rust SVD (OpenBLAS) and NumPy SVD (platform LAPACK) can give
    # slightly different principal components, so we allow some slack.
    _COS_TOL = 0.95

    def test_he_like_image_stain_alignment(self, he_like_image):
        """Estimated stains should be close to the known H&E directions."""
        w_ours = rgb_separate_stains_macenko_pca(he_like_image)
        w_ref = _htk_separate_stains_macenko_pca(
            _htk_rgb_to_sda(he_like_image),
        )

        for name in ("hematoxylin", "eosin"):
            ref_vec = np.array(stain_color_map[name], dtype=np.float64)

            idx_ours = find_stain_index(ref_vec, w_ours)
            idx_ref = find_stain_index(ref_vec, w_ref)

            col_ours = w_ours[:, idx_ours].astype(np.float64)
            col_ref = w_ref[:, idx_ref].astype(np.float64)

            cos_sim = abs(
                np.dot(col_ours, col_ref)
                / (np.linalg.norm(col_ours) * np.linalg.norm(col_ref))
            )
            assert cos_sim > self._COS_TOL, (
                f"{name}: cosine similarity {cos_sim:.4f} < {self._COS_TOL}"
            )

    def test_he_like_image_columns_normalised(self, he_like_image):
        w = rgb_separate_stains_macenko_pca(he_like_image)
        norms = np.linalg.norm(w.astype(np.float64), axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_he_like_image_complement_orthogonal(self, he_like_image):
        """Third column should be orthogonal to the first two."""
        w = rgb_separate_stains_macenko_pca(he_like_image).astype(np.float64)
        assert abs(np.dot(w[:, 0], w[:, 2])) < 1e-6
        assert abs(np.dot(w[:, 1], w[:, 2])) < 1e-6

    def test_f32_vs_f64_stain_alignment(self, he_like_image):
        """f32 and f64 pipelines should produce similar stain vectors."""
        w64 = rgb_separate_stains_macenko_pca(he_like_image)
        w32 = rgb_separate_stains_macenko_pca(he_like_image.astype(np.float32))

        for name in ("hematoxylin", "eosin"):
            ref_vec = np.array(stain_color_map[name], dtype=np.float64)
            i64 = find_stain_index(ref_vec, w64)
            i32 = find_stain_index(ref_vec, w32)
            c64 = w64[:, i64].astype(np.float64)
            c32 = w32[:, i32].astype(np.float64)
            cos_sim = abs(
                np.dot(c64, c32) / (np.linalg.norm(c64) * np.linalg.norm(c32))
            )
            assert cos_sim > 0.99, f"{name}: f32/f64 cosine similarity {cos_sim:.4f}"

    def test_sda_pathway_matches_rgb_pathway(self, he_like_image):
        """separate_stains_macenko_pca(sda) should match rgb_separate_stains."""
        w_rgb = rgb_separate_stains_macenko_pca(he_like_image)
        sda = rgb_to_sda(he_like_image)
        w_sda = separate_stains_macenko_pca(sda)
        np.testing.assert_allclose(w_rgb, w_sda, atol=1e-10)

    def test_reference_close_to_known_vectors(self, he_like_image):
        """Both implementations should find vectors close to known H&E."""
        w = rgb_separate_stains_macenko_pca(he_like_image)

        for name in ("hematoxylin", "eosin"):
            ref_vec = np.array(stain_color_map[name], dtype=np.float64)
            ref_vec = ref_vec / np.linalg.norm(ref_vec)
            idx = find_stain_index(ref_vec, w)
            col = w[:, idx].astype(np.float64)
            cos_sim = abs(np.dot(ref_vec, col) / np.linalg.norm(col))
            assert cos_sim > 0.8, (
                f"{name}: cosine to known reference only {cos_sim:.4f}"
            )

    def test_random_image_doesnt_crash(self, sample_rgb):
        """Even random images should produce a valid 3x3 stain matrix."""
        w = rgb_separate_stains_macenko_pca(sample_rgb)
        assert w.shape == (3, 3)
        assert np.all(np.isfinite(w))
        norms = np.linalg.norm(w.astype(np.float64), axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_with_mask_parity(self, he_like_image):
        """Masked estimation should still produce aligned stain vectors."""
        mask = np.zeros(he_like_image.shape[:2], dtype=bool)
        mask[:10, :] = True  # mask out top 10 rows

        w_masked = rgb_separate_stains_macenko_pca(he_like_image, mask_out=mask)
        w_full = rgb_separate_stains_macenko_pca(he_like_image)

        for name in ("hematoxylin", "eosin"):
            ref_vec = np.array(stain_color_map[name], dtype=np.float64)
            i_m = find_stain_index(ref_vec, w_masked)
            i_f = find_stain_index(ref_vec, w_full)
            cm = w_masked[:, i_m].astype(np.float64)
            cf = w_full[:, i_f].astype(np.float64)
            cos_sim = abs(np.dot(cm, cf) / (np.linalg.norm(cm) * np.linalg.norm(cf)))
            # Masking 10 rows out of 128 shouldn't change things much
            assert cos_sim > 0.9, (
                f"{name}: masked vs full cosine similarity {cos_sim:.4f}"
            )


# =====================================================================
# End-to-end pipeline parity
# =====================================================================


class TestEndToEndPipelineParity:
    """Full pipeline comparisons: our Rust backend vs NumPy reference."""

    def test_sda_deconvolution_reconstruction_roundtrip_f64(
        self, sample_rgb, known_stain_matrix
    ):
        """RGB -> SDA -> concentrations -> reconstruct should approximate original."""
        conc_ours = rgb_color_deconvolution(sample_rgb, known_stain_matrix)
        recon_ours = reconstruct_rgb(conc_ours, known_stain_matrix)

        sda_ref = _htk_rgb_to_sda(sample_rgb)
        conc_ref = _htk_color_deconvolution(sda_ref, known_stain_matrix)
        recon_ref = _htk_reconstruct_rgb(conc_ref, known_stain_matrix)

        # Concentrations should be very close
        np.testing.assert_allclose(conc_ours, conc_ref, atol=1e-8, rtol=1e-8)
        # Reconstructions should be very close
        np.testing.assert_allclose(recon_ours, recon_ref, atol=1e-6, rtol=1e-6)

    def test_estimated_stain_deconvolution_parity(self, he_like_image):
        """Use *our* estimated stain matrix, then compare deconvolution.

        Since the stain matrix is the same (ours), the deconvolution and
        reconstruction should match the reference math exactly.
        """
        w = rgb_separate_stains_macenko_pca(he_like_image)
        w64 = w.astype(np.float64)

        conc_ours = rgb_color_deconvolution(he_like_image, w)
        sda_ref = _htk_rgb_to_sda(he_like_image)
        conc_ref = _htk_color_deconvolution(sda_ref, w64)
        np.testing.assert_allclose(conc_ours, conc_ref, atol=1e-8, rtol=1e-8)

        recon_ours = reconstruct_rgb(conc_ours, w)
        recon_ref = _htk_reconstruct_rgb(conc_ref, w64)
        np.testing.assert_allclose(recon_ours, recon_ref, atol=1e-6, rtol=1e-6)

    def test_stain_isolation_parity(self, he_like_image):
        """Isolating stain 0 via our pipeline should match the reference."""
        w = rgb_separate_stains_macenko_pca(he_like_image)
        w64 = w.astype(np.float64)

        conc = rgb_color_deconvolution(he_like_image, w)
        conc_s0 = conc.copy()
        conc_s0[:, :, 1] = 0.0
        conc_s0[:, :, 2] = 0.0

        recon_ours = reconstruct_rgb(conc_s0, w)
        recon_ref = _htk_reconstruct_rgb(conc_s0.astype(np.float64), w64)
        np.testing.assert_allclose(recon_ours, recon_ref, atol=1e-6, rtol=1e-6)

    def test_full_pipeline_roundtrip_close_to_original_f64(self, he_like_image):
        """Estimate stains -> deconvolve -> reconstruct ~ original (f64)."""
        w = rgb_separate_stains_macenko_pca(he_like_image)
        conc = rgb_color_deconvolution(he_like_image, w)
        recon = reconstruct_rgb(conc, w)
        # Lossy due to SDA clamping, but should be close
        np.testing.assert_allclose(recon, he_like_image, atol=5.0, rtol=0.05)

    def test_full_pipeline_roundtrip_close_to_original_f32(self, he_like_image):
        """Estimate stains -> deconvolve -> reconstruct ~ original (f32)."""
        im32 = he_like_image.astype(np.float32)
        w = rgb_separate_stains_macenko_pca(im32)
        conc = rgb_color_deconvolution(im32, w)
        recon = reconstruct_rgb(conc, w)
        np.testing.assert_allclose(recon, im32, atol=5.0, rtol=0.05)

    def test_find_stain_index_consistent_after_deconvolution(self, he_like_image):
        """After deconvolution, the stain index assignment should be stable."""
        w = rgb_separate_stains_macenko_pca(he_like_image)
        h_idx = find_stain_index(stain_color_map["hematoxylin"], w)
        e_idx = find_stain_index(stain_color_map["eosin"], w)

        # H and E should map to different columns in {0, 1}
        assert h_idx != e_idx
        assert {h_idx, e_idx} == {0, 1}

        # The two real stain channels should be mostly non-negative; the
        # third (cross-product complement) channel can go negative freely
        # because it captures residual signal, not a real stain.
        conc = rgb_color_deconvolution(he_like_image, w)
        stain_conc = conc[:, :, [h_idx, e_idx]]
        negative_fraction = np.mean(stain_conc < -0.1)
        assert negative_fraction < 0.05, (
            f"Too many negative concentrations in stain channels: "
            f"{negative_fraction:.2%}"
        )

    def test_different_stains_produce_different_concentrations(self, he_like_image):
        """Columns identified as H vs E should have different concentration maps."""
        w = rgb_separate_stains_macenko_pca(he_like_image)
        conc = rgb_color_deconvolution(he_like_image, w)
        h_idx = find_stain_index(stain_color_map["hematoxylin"], w)
        e_idx = find_stain_index(stain_color_map["eosin"], w)
        h_conc = conc[:, :, h_idx]
        e_conc = conc[:, :, e_idx]
        # They should not be identical
        assert not np.allclose(h_conc, e_conc, atol=0.1)

    def test_multiple_images_consistent(self, rng):
        """Pipeline should be consistent across multiple synthetic images."""
        for seed in range(5):
            local_rng = np.random.default_rng(seed + 100)
            h = np.array([0.65, 0.70, 0.29])
            e = np.array([0.07, 0.99, 0.11])
            n = 64 * 64
            c_h = local_rng.uniform(0.1, 1.0, n)
            c_e = local_rng.uniform(0.1, 1.0, n)
            od = np.outer(c_h, h) + np.outer(c_e, e)
            od += local_rng.normal(0, 0.02, od.shape)
            od = np.clip(od, 0, None)
            rgb = np.clip(256.0 * np.exp(-od), 1.0, 255.0).reshape(64, 64, 3)

            w = rgb_separate_stains_macenko_pca(rgb)

            # Should always produce valid stain matrix
            assert w.shape == (3, 3)
            assert np.all(np.isfinite(w))

            # H and E should map to distinct columns in {0, 1}
            h_idx = find_stain_index(stain_color_map["hematoxylin"], w)
            e_idx = find_stain_index(stain_color_map["eosin"], w)
            assert h_idx != e_idx, f"seed={seed}: H and E mapped to same column"
            assert {h_idx, e_idx} == {0, 1}, (
                f"seed={seed}: expected {{0,1}}, got {{{h_idx},{e_idx}}}"
            )

            # Deconvolution should work
            conc = rgb_color_deconvolution(rgb, w)
            assert np.all(np.isfinite(conc))

            # Roundtrip should be close
            recon = reconstruct_rgb(conc, w)
            np.testing.assert_allclose(recon, rgb, atol=5.0, rtol=0.05)
