use crate::utils::create_pool;
use indexmap::IndexMap;
use ndarray::{Array1, Array2, ArrayView1, Zip};
use num::{one, zero, PrimInt, Zero};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyFixedString, PyReadonlyArray1};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use rayon::{prelude::ParallelIterator, ThreadPoolBuildError};
use std::{fmt::Display, hash::Hash, ops::AddAssign};

type CoordInt = i32;
type CodeInt = i32;

#[pyfunction]
#[pyo3(signature = (x, y, *, n_threads=None))]
/// Concatenate two int arrays into a string separated by underscore
pub fn coordinate_as_string<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, CoordInt>,
    y: PyReadonlyArray1<'py, CoordInt>,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<PyFixedString<12>>>> {
    match string_coordinate_index_(x.as_array(), y.as_array(), n_threads) {
        Ok(string_coordinates) => Ok(string_coordinates
            .map(|s| (*s).into())
            .into_pyarray_bound(py)),
        Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
    }
}

#[pyfunction]
#[pyo3(signature = (x, y, *, n_threads=None))]
/// From a list of coordinates extract a categorical representation
pub fn categorical_coordinate<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, CoordInt>,
    y: PyReadonlyArray1<'py, CoordInt>,
    n_threads: Option<usize>,
) -> PyResult<(
    Bound<'py, PyArray1<CodeInt>>,
    Bound<'py, PyArray2<CoordInt>>,
)> {
    match categorical_coordinate_(x.as_array(), y.as_array(), n_threads) {
        Ok((codes, coordinates)) => Ok((
            codes.into_pyarray_bound(py),
            coordinates.into_pyarray_bound(py),
        )),
        Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
    }
}

//// pure Rust part

/// Concatenate two int arrays into a 'string' array
fn string_coordinate_index_<'a, X, const N: usize>(
    x: ArrayView1<'a, X>,
    y: ArrayView1<'a, X>,
    n_threads: Option<usize>,
) -> Result<Array1<[u8; N]>, ThreadPoolBuildError>
where
    X: Display,
    &'a X: Send,
    ArrayView1<'a, X>: Sync + Send,
{
    let thread_pool = create_pool(n_threads)?;

    let mut x_y = Array1::from_elem(x.len(), [0u8; N]);
    thread_pool.install(|| {
        Zip::from(&mut x_y).and(x).and(y).par_for_each(|xy, x, y| {
            let mut xy_string = String::with_capacity(N);
            xy_string.push_str(&x.to_string());
            xy_string.push('_');
            xy_string.push_str(&y.to_string());

            let xy_bytes = xy_string.as_bytes();
            (*xy)[..xy_bytes.len()].copy_from_slice(xy_bytes);
        });
    });

    Ok(x_y)
}

fn categorical_coordinate_<'a, C, X>(
    x: ArrayView1<'a, X>,
    y: ArrayView1<'a, X>,
    n_threads: Option<usize>,
) -> Result<(Array1<C>, Array2<X>), ThreadPoolBuildError>
where
    C: PrimInt + Sync + Send + AddAssign,
    X: Copy + Zero + Sync + Send,
    (X, X): Eq + PartialEq + Hash,
    &'a X: Send,
    ArrayView1<'a, X>: Sync + Send,
{
    let thread_pool = create_pool(n_threads)?;

    let n = x.len();
    let n_coord_estimate = n / 5; // rough guess of size to reduce allocations

    let mut coord2idx: IndexMap<(X, X), C> = IndexMap::with_capacity(n_coord_estimate);

    {
        let mut cnt = zero::<C>();
        Zip::from(x).and(y).for_each(|x, y| {
            coord2idx.entry((*x, *y)).or_insert_with(|| {
                let curr = cnt;
                cnt += one();
                curr
            });
        });
    }

    let codes = thread_pool.install(|| {
        Zip::from(x).and(y).par_map_collect(|x, y| {
            *coord2idx
                .get(&(*x, *y))
                .expect("All coordinates are initialized in HashMap")
        })
    });

    let coordinates = thread_pool.install(|| {
        coord2idx
            .par_keys()
            .map(|row| [row.0, row.1])
            .collect::<Vec<_>>()
            .into()
    });

    Ok((codes, coordinates))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_string_coordinate() {
        let a = array![0, 1, 99_999];
        let b = array![0, 20, 99_999];

        let a_b: Array1<[u8; 12]> = string_coordinate_index_(a.view(), b.view(), None).unwrap();

        let a_b_string: Vec<[u8; 12]> = vec![
            [48, 95, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [49, 95, 50, 48, 0, 0, 0, 0, 0, 0, 0, 0],
            [57, 57, 57, 57, 57, 95, 57, 57, 57, 57, 57, 0],
        ];
        assert_eq!(a_b, Array1::from_vec(a_b_string));
    }

    #[test]
    fn test_categorical_coordinate() {
        let a = array![0, 0, 1, 0, 1];
        let b = array![0, 1, 0, 0, 1];

        let (codes, coord): (Array1<i32>, Array2<i32>) =
            categorical_coordinate_(a.view(), b.view(), None).unwrap();

        let codes_test = array![0, 1, 2, 0, 3];
        let coord_test = array![[0, 0], [0, 1], [1, 0], [1, 1]];

        assert_eq!(codes, codes_test);
        assert_eq!(coord, coord_test);
    }
}
