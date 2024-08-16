use crate::{
    gridcounts::{Count, CsxIndex, GridCounts},
    sparsearray_conversion::WrappedCsx,
    utils::create_pool,
};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, NdFloat, NewAxis, Slice, Zip};
use num::{one, zero, NumCast, PrimInt, Signed, Zero};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use rayon::{iter::ParallelIterator, prelude::ParallelSlice, ThreadPoolBuildError};
use sprs::{hstack, CsMatBase, CsMatI, CsMatViewI, SpIndex};
use std::{
    cmp::{max, min},
    ops::{Deref, Range},
};

type KDEPrecision = f32;

macro_rules! build_kde_csx_fn {
    ($name:tt, $t_count:ty, $t_index:ty) => {
        #[pyfunction]
        #[pyo3(signature = (counts, kernel, *, threshold=0.0))]
        /// calculate sparse KDE
        pub fn $name<'py>(
            _py: Python<'py>,
            counts: WrappedCsx<$t_count, $t_index, $t_index>,
            kernel: PyReadonlyArray2<'py, KDEPrecision>,
            threshold: KDEPrecision,
        ) -> PyResult<WrappedCsx<KDEPrecision, $t_index, $t_index>> {
            let sparse_kde = sparse_kde_csx(&counts.0, kernel.as_array(), threshold);
            Ok(WrappedCsx(sparse_kde))
        }
    };
}

build_kde_csx_fn!(sparse_kde_csx_py, Count, CsxIndex);

#[pyfunction]
#[pyo3(signature = (counts, genes, kernel, coordinates, *, n_threads=None))]
/// calculate KDE and retrieve coordinates
pub fn kde_at_coord<'py>(
    _py: Python<'py>,
    counts: &GridCounts,
    genes: Vec<String>,
    kernel: PyReadonlyArray2<'py, KDEPrecision>,
    coordinates: (PyReadonlyArray1<'py, isize>, PyReadonlyArray1<'py, isize>),
    n_threads: Option<usize>,
) -> PyResult<WrappedCsx<KDEPrecision, usize, usize>> {
    let gene_counts: Vec<_> = genes
        .iter()
        .map(|g| {
            counts
                .get_view(g)
                .ok_or(PyValueError::new_err("Not all genes exist"))
        })
        .collect::<Result<_, _>>()?;

    let coordinates = (coordinates.0.as_array(), coordinates.1.as_array());

    match kde_at_coord_(&gene_counts, kernel.as_array(), coordinates, n_threads) {
        Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
        Ok(kde_coord) => Ok(WrappedCsx(kde_coord)),
    }
}

#[inline]
fn in_bounds_range<I: Signed + PrimInt>(n: I, i: I, pad: I) -> (I, I) {
    (max(-i, -pad), min(n - i, pad + one()))
}

fn sparse_kde_csx<C, I, Iptr, F>(
    counts: &CsMatI<C, I, Iptr>,
    kernel: ArrayView2<F>,
    threshold: F,
) -> CsMatI<F, I>
where
    C: NumCast + Copy,
    I: SpIndex + Signed,
    Iptr: SpIndex,
    F: NdFloat + Signed,
    Slice: From<Range<I>>,
{
    let mut kde = Array2::zeros(counts.shape());
    sparse_kde_csx_(&mut kde, counts, kernel);
    CsMatI::csr_from_dense(ArrayView2::from(&kde), threshold)
}

pub fn sparse_kde_csx_<C, I, Iptr, IptrStorage, IndStorage, DataStorage, F>(
    kde: &mut Array2<F>,
    counts: &CsMatBase<C, I, IptrStorage, IndStorage, DataStorage, Iptr>,
    kernel: ArrayView2<F>,
) where
    C: NumCast + Copy,
    I: SpIndex + Signed,
    Iptr: SpIndex,
    F: NdFloat,
    Slice: From<Range<I>>,
    IptrStorage: Deref<Target = [Iptr]>,
    IndStorage: Deref<Target = [I]>,
    DataStorage: Deref<Target = [C]>,
{
    let shape = kde.shape();
    let (m, n) = (I::from(shape[0]).unwrap(), I::from(shape[1]).unwrap());

    let shift_i = I::from((kernel.nrows() - 1) / 2).unwrap();
    let shift_j = I::from((kernel.ncols() - 1) / 2).unwrap();

    kde.fill(zero());

    counts.iter().for_each(|(&val, (i, j))| {
        let (i_min, i_max) = in_bounds_range(m, i, shift_i);
        let (j_min, j_max) = in_bounds_range(n, j, shift_j);
        let val = F::from(val).unwrap();

        Zip::from(&mut kde.slice_mut(s![
            Slice::from((i + i_min)..(i + i_max)),
            Slice::from((j + j_min)..(j + j_max))
        ]))
        .and(&kernel.slice(s![
            Slice::from((shift_i + i_min)..(shift_i + i_max)),
            Slice::from((shift_j + j_min)..(shift_j + j_max))
        ]))
        .for_each(|kde, &k| {
            *kde += k * val;
        });
    })
}

fn kde_at_coord_<C, I, F, I2>(
    counts: &[CsMatViewI<C, I>],
    kernel: ArrayView2<F>,
    coordinates: (ArrayView1<I2>, ArrayView1<I2>),
    n_threads: Option<usize>,
) -> Result<CsMatI<F, usize>, ThreadPoolBuildError>
where
    C: NumCast + Copy + Sync + Send,
    I: SpIndex + Signed + Sync + Send,
    I2: PrimInt,
    F: NdFloat + Signed + Default,
    Slice: From<Range<I>>,
{
    let pool = create_pool(n_threads)?;
    let n_threads = pool.current_num_threads();

    let shape = counts.first().expect("At least one gene").shape();

    let coord_x = coordinates
        .0
        .mapv(|x| <usize as NumCast>::from(x).unwrap())
        .to_vec();
    let coord_y = coordinates
        .1
        .mapv(|x| <usize as NumCast>::from(x).unwrap())
        .to_vec();

    let batch_size = counts.len().div_ceil(n_threads);

    let kde_coords = pool.install(|| {
        counts
            .par_chunks(batch_size)
            .map(|counts_batch| {
                let mut kde_buffer = Array2::zeros(shape);
                let mut kde_coords_batch = Vec::with_capacity(counts_batch.len());
                for c in counts_batch {
                    sparse_kde_csx_(&mut kde_buffer, c, kernel);
                    kde_coords_batch.push(get_coord(kde_buffer.view(), (&coord_x, &coord_y)));
                }
                kde_coords_batch
            })
            .flatten_iter()
            .collect::<Vec<_>>()
    });

    Ok(hstack(
        &kde_coords.iter().map(|x| x.view()).collect::<Vec<_>>(),
    ))
}

fn get_coord<T: Zero + Copy + PartialOrd + Signed>(
    arr: ArrayView2<T>,
    coordinates: (&[usize], &[usize]),
) -> CsMatI<T, usize> {
    let mut out = Array1::zeros(coordinates.0.len());

    Zip::from(&mut out)
        .and(coordinates.0)
        .and(coordinates.1)
        .for_each(|o, &i, &j| *o = arr[[i, j]]);
    CsMatI::csc_from_dense(out.slice(s![.., NewAxis]), zero())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    struct Setup {
        counts: CsMatI<i32, i32>,
        kernel: Array2<f64>,
        threshold: f64,
        kde: CsMatI<f64, i32, i32>,
    }

    impl Setup {
        fn new() -> Self {
            Self {
                counts: CsMatI::new(
                    (6, 12),
                    vec![0, 1, 2, 3, 3, 3, 4],
                    vec![0, 2, 3, 10],
                    vec![1, 1, 2, 1],
                ),
                kernel: array![[0.5, 0.0, 0.0], [0.5, 1.0, 0.0], [0.25, 0.0, 0.0]],
                threshold: 0.4,
                kde: CsMatI::new(
                    (6, 12),
                    vec![0, 2, 4, 6, 7, 8, 10],
                    vec![0, 1, 1, 2, 2, 3, 2, 9, 9, 10],
                    vec![1.0, 0.5, 0.5, 2.0, 1.0, 2.0, 0.5, 0.5, 0.5, 1.0],
                ),
            }
        }
    }

    // Input looks like this
    // [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    //  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    //  [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    //  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    //  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    //  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]

    // Unfiltered output should look like this but after thresholding the 0.25 is removed
    // [[1.0, 0.5 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //  [0.0, 0.5 , 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //  [0.0, 0.25, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //  [0.0, 0.0 , 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //  [0.0, 0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
    //  [0.0, 0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0]]

    // let kde_test = array![
    //     [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //     [0.0, 0.5, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //     [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //     [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    //     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
    //     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0]
    // ];

    #[test]
    fn test_sparse_kde_csx() {
        let setup = Setup::new();

        let kde = sparse_kde_csx(&setup.counts, setup.kernel.view(), setup.threshold);

        assert_eq!(kde, setup.kde);
    }

    #[test]
    fn test_get_coord() {
        let coord_x = vec![0, 1, 1];
        let coord_y = vec![0, 2, 3];

        let setup = Setup::new();

        let kde_coord = get_coord(setup.kde.to_dense().view(), (&coord_x, &coord_y));

        let result = CsMatI::new_csc((3, 1), vec![0, 2], vec![0, 1], vec![1., 2.]);

        assert_eq!(kde_coord, result)
    }
    #[test]
    fn test_kde_at_coord() {
        let coord_x = array![0, 1, 1];
        let coord_y = array![0, 2, 3];

        let setup = Setup::new();

        let counts = vec![setup.counts.view(), setup.counts.view()];

        let kde_coord = kde_at_coord_(
            &counts,
            setup.kernel.view(),
            (coord_x.view(), coord_y.view()),
            None,
        )
        .unwrap();

        let result = CsMatI::new_csc(
            (3, 2),
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1., 2., 1., 2.],
        );

        assert_eq!(kde_coord, result)
    }
}
