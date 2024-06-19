mod coordinates;
mod cosine;
mod gridcounts;
mod sparsearray_conversion;
mod sparsekde;
mod utils;

use coordinates::{categorical_coordinate, coordinate_as_string};
use cosine::{cosinef32_and_celltypei16, cosinef32_and_celltypei8};
use gridcounts::GridCounts;
use pyo3::prelude::*;
use sparsekde::{kde_at_coord, sparse_kde_csx_py};

/// utils written in Rust to improve performance
#[pymodule]
// #[pyo3(name = "_utils_rust")]
fn _utils_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GridCounts>()?;
    m.add_function(wrap_pyfunction!(sparse_kde_csx_py, m)?)?;
    m.add_function(wrap_pyfunction!(kde_at_coord, m)?)?;
    m.add_function(wrap_pyfunction!(cosinef32_and_celltypei8, m)?)?;
    m.add_function(wrap_pyfunction!(cosinef32_and_celltypei16, m)?)?;
    m.add_function(wrap_pyfunction!(coordinate_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(categorical_coordinate, m)?)?;
    Ok(())
}
