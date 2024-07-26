mod coordinates;
mod cosine;
mod gridcounts;
mod sparsearray_conversion;
mod sparsekde;
mod utils;

use coordinates::{categorical_coordinate, coordinate_as_string};
use cosine::{
    gridcounts_cosinef32_celltypei16, gridcounts_cosinef32_celltypei8,
    gridfloats_cosinef32_celltypei16, gridfloats_cosinef32_celltypei8,
};
use gridcounts::GridCounts;
use gridcounts::GridFloats;
use pyo3::prelude::*;
use sparsekde::{
    gridcounts_kde_at_coord, gridfloats_kde_at_coord, sparse_kde_csxf32, sparse_kde_csxu32,
};

/// utils written in Rust to improve performance
#[pymodule]
// #[pyo3(name = "_utils_rust")]
fn _utils_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GridCounts>()?;
    m.add_class::<GridFloats>()?;
    m.add_function(wrap_pyfunction!(sparse_kde_csxf32, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_kde_csxu32, m)?)?;
    m.add_function(wrap_pyfunction!(gridcounts_kde_at_coord, m)?)?;
    m.add_function(wrap_pyfunction!(gridfloats_kde_at_coord, m)?)?;
    m.add_function(wrap_pyfunction!(gridcounts_cosinef32_celltypei16, m)?)?;
    m.add_function(wrap_pyfunction!(gridcounts_cosinef32_celltypei8, m)?)?;
    m.add_function(wrap_pyfunction!(gridfloats_cosinef32_celltypei16, m)?)?;
    m.add_function(wrap_pyfunction!(gridfloats_cosinef32_celltypei8, m)?)?;
    m.add_function(wrap_pyfunction!(coordinate_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(categorical_coordinate, m)?)?;
    Ok(())
}
