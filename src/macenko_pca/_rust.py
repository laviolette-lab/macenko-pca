"""
Lightweight stub for the compiled native extension.

The real package provides a compiled extension module named
`macenko_pca._rust` implemented in Rust (via PyO3). That extension
exports several high-performance functions which are imported by the
pure-Python wrappers in `macenko_pca.deconvolution`.

This file is a thin, importable fallback used for static analysis,
documentation builds and developer workflows where the native extension
has not been built. The stubs expose the same symbols (function names)
and signatures (accepting arbitrary args/kwargs) but raise a clear and
actionable RuntimeError if called at runtime.

Do not rely on these stubs for production use. Build and install the
native extension to get the implemented functionality.
"""

from typing import Any, Callable

# Public API names exported by the real native module.
__all__ = [
    "py_color_deconvolution_f32",
    "py_color_deconvolution_f64",
    "py_reconstruct_rgb_f32",
    "py_reconstruct_rgb_f64",
    "py_rgb_color_deconvolution_f32",
    "py_rgb_color_deconvolution_f64",
    "py_rgb_separate_stains_macenko_pca_f32",
    "py_rgb_separate_stains_macenko_pca_f64",
    "py_rgb_to_sda_f32",
    "py_rgb_to_sda_f64",
    "py_separate_stains_macenko_pca_f32",
    "py_separate_stains_macenko_pca_f64",
]


def _unavailable(name: str) -> Callable[..., Any]:
    """Return a callable that raises a RuntimeError explaining the situation.

    The returned callable accepts arbitrary arguments to match the native
    functions' flexible signatures, but will always raise when invoked.
    """

    def _fn(
        *args: Any, **kwargs: Any
    ) -> Any:  # pragma: no cover - exercised only when native ext missing
        raise RuntimeError(
            "Native extension 'macenko_pca._rust' is not available. "
            f"Attempted to call '{name}'. Build and install the package "
            "with the native extension (for example, using 'maturin develop' "
            "or following the project's README) to enable this function."
        )

    return _fn


# Stubs for the functions exported by the compiled extension.
# Each one mirrors the exported symbol name so `from ... import ...`
# and attribute access in the Python wrappers succeed at import time.
py_color_deconvolution_f32 = _unavailable("py_color_deconvolution_f32")
py_color_deconvolution_f64 = _unavailable("py_color_deconvolution_f64")
py_reconstruct_rgb_f32 = _unavailable("py_reconstruct_rgb_f32")
py_reconstruct_rgb_f64 = _unavailable("py_reconstruct_rgb_f64")
py_rgb_color_deconvolution_f32 = _unavailable("py_rgb_color_deconvolution_f32")
py_rgb_color_deconvolution_f64 = _unavailable("py_rgb_color_deconvolution_f64")
py_rgb_separate_stains_macenko_pca_f32 = _unavailable(
    "py_rgb_separate_stains_macenko_pca_f32"
)
py_rgb_separate_stains_macenko_pca_f64 = _unavailable(
    "py_rgb_separate_stains_macenko_pca_f64"
)
py_rgb_to_sda_f32 = _unavailable("py_rgb_to_sda_f32")
py_rgb_to_sda_f64 = _unavailable("py_rgb_to_sda_f64")
py_separate_stains_macenko_pca_f32 = _unavailable("py_separate_stains_macenko_pca_f32")
py_separate_stains_macenko_pca_f64 = _unavailable("py_separate_stains_macenko_pca_f64")
