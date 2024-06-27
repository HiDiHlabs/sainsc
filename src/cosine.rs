use crate::gridcounts::GridCounts;
use crate::sparsekde::sparse_kde_csx_;
use crate::utils::create_pool;

use ndarray::{
    concatenate, s, Array2, Array3, ArrayView1, ArrayView2, Axis, NdFloat, NewAxis, ShapeError,
    Slice, Zip,
};
use num::{one, zero, NumCast, PrimInt, Signed, Zero};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use sprs::{CompressedStorage::CSR, CsMatI, CsMatViewI, SpIndex};
use std::{
    cmp::{max, min},
    error::Error,
    ops::{Range, Sub},
};

macro_rules! build_cos_ct_fn {
    ($name:tt, $t_cos:ty, $t_ct:ty) => {
        #[pyfunction]
        #[pyo3(signature = (counts, genes, signatures, kernel, *, log=false, chunk_size=(500, 500), n_threads=None))]
        /// calculate cosine similarity and assign celltype
        pub fn $name<'py>(
            py: Python<'py>,
            counts: &mut GridCounts,
            genes: Vec<String>,
            signatures: PyReadonlyArray2<'py, $t_cos>,
            kernel: PyReadonlyArray2<'py, $t_cos>,
            log: bool,
            chunk_size: (usize, usize),
            n_threads: Option<usize>,
        ) -> PyResult<(Bound<'py, PyArray2<$t_cos>>, Bound<'py, PyArray2<$t_cos>>, Bound<'py, PyArray2<$t_ct>>)> {

            // ensure that all count arrays are CSR
            counts.to_format(CSR);
            let gene_counts: Vec<_> = genes
                .iter()
                .map(|g| {
                    counts
                        .get_view(g)
                        .ok_or(PyValueError::new_err("Not all genes exist"))
                })
                .collect::<Result<_, _>>()?;

            let cos_ct = chunk_and_calculate_cosine(
                &gene_counts,
                signatures.as_array(),
                kernel.as_array(),
                counts.shape,
                log,
                chunk_size,
                n_threads.unwrap_or(0),
            );

            match cos_ct {
                Ok((cosine, score, celltype_map)) => Ok((
                    cosine.into_pyarray_bound(py),
                    score.into_pyarray_bound(py),
                    celltype_map.into_pyarray_bound(py),
                )),
                Err(e) => Err(PyValueError::new_err(e.to_string())),
            }
        }
    };
}

build_cos_ct_fn!(cosinef32_and_celltypei8, f32, i8);
build_cos_ct_fn!(cosinef32_and_celltypei16, f32, i16);

fn chunk_and_calculate_cosine<'a, C, I, F, U>(
    counts: &[CsMatViewI<'a, C, I>],
    signatures: ArrayView2<'a, F>,
    kernel: ArrayView2<'a, F>,
    shape: (usize, usize),
    log: bool,
    chunk_size: (usize, usize),
    n_threads: usize,
) -> Result<(Array2<F>, Array2<F>, Array2<U>), Box<dyn Error>>
where
    C: NumCast + Copy + Sync + Send + Default,
    I: SpIndex + Signed + Sync + Send,
    F: NdFloat,
    U: PrimInt + Signed + Sync + Send,
    Slice: From<Range<I>>,
{
    let pool = create_pool(n_threads)?;

    let kernelsize = kernel.shape();

    let (nrow, ncol) = shape;
    let (srow, scol) = chunk_size;
    let (padrow, padcol) = ((kernelsize[0] - 1) / 2, (kernelsize[1] - 1) / 2);
    let (m, n) = (nrow.div_ceil(srow), ncol.div_ceil(scol)); // number of chunks

    let n_celltype = signatures.ncols();
    let signature_similarity_correction =
        Array2::from_shape_fn((n_celltype, n_celltype), |(i, j)| {
            if i > j {
                let sig1 = signatures.index_axis(Axis(1), i);
                let sig2 = signatures.index_axis(Axis(1), j);
                // The dot product calculates the cosine similarity (signatures are normalized)
                // this corresponds to cos(x) but we want cos(pi/2 - x) = sin(x)
                // -> sin(acos(dot_product))
                sig1.dot(&sig2).acos().sin()
            } else {
                zero()
            }
        });

    let mut cosine_rows = Vec::with_capacity(m);
    let mut score_rows = Vec::with_capacity(m);
    let mut celltype_rows = Vec::with_capacity(m);

    pool.install(|| {
        for i in 0..m {
            let (slice_row, unpad_row) = chunk_(i, srow, nrow, padrow);
            let row_chunk: Vec<_> = counts
                .par_iter()
                .map(|c| {
                    c.slice_outer(slice_row.clone())
                        .transpose_view()
                        .to_other_storage()
                })
                .collect();

            let ((cosine_cols, score_cols), celltype_cols): (
                (Vec<Array2<F>>, Vec<Array2<F>>),
                Vec<Array2<U>>,
            ) = (0..n)
                .into_par_iter()
                .map(|j| {
                    let (slice_col, unpad_col) = chunk_(j, scol, ncol, padcol);

                    let chunk = row_chunk
                        .par_iter()
                        .map(|c| c.slice_outer(slice_col.clone()).transpose_into().to_owned())
                        .collect();

                    cosine_and_celltype_(
                        chunk,
                        signatures,
                        signature_similarity_correction.view(),
                        kernel,
                        (unpad_row.clone(), unpad_col),
                        log,
                    )
                })
                .unzip();
            cosine_rows.push(concat_1d(cosine_cols, 1));
            score_rows.push(concat_1d(score_cols, 1));
            celltype_rows.push(concat_1d(celltype_cols, 1));
        }
    });

    let cosine = concat_1d(cosine_rows.into_iter().collect::<Result<Vec<_>, _>>()?, 0)?;
    let score = concat_1d(score_rows.into_iter().collect::<Result<Vec<_>, _>>()?, 0)?;
    let celltype = concat_1d(celltype_rows.into_iter().collect::<Result<Vec<_>, _>>()?, 0)?;
    Ok((cosine, score, celltype))
}

fn concat_1d<T: Clone + Sync + Send>(
    chunks: Vec<Array2<T>>,
    axis: usize,
) -> Result<Array2<T>, ShapeError> {
    concatenate(
        Axis(axis),
        &chunks.par_iter().map(|a| a.view()).collect::<Vec<_>>(),
    )
}

fn chunk_(i: usize, step: usize, n: usize, pad: usize) -> (Range<usize>, Range<usize>) {
    let bound1 = (i * step) as isize;
    let bound2 = (i + 1) * step;
    let start = max(0, bound1 - pad as isize) as usize;
    let start2 = max(0, bound1 - start as isize) as usize;
    let chunk_pad = start..min(n, bound2 + pad);
    let chunk_unpad = start2..(start2 + min(step, (n as isize - bound1) as usize));
    (chunk_pad, chunk_unpad)
}

fn cosine_and_celltype_<'a, C, I, F, U>(
    counts: Vec<CsMatI<C, I>>,
    signatures: ArrayView2<'a, F>,
    pairwise_correction: ArrayView2<F>,
    kernel: ArrayView2<'a, F>,
    unpad: (Range<usize>, Range<usize>),
    log: bool,
) -> ((Array2<F>, Array2<F>), Array2<U>)
where
    C: NumCast + Copy,
    F: NdFloat,
    U: PrimInt + Signed,
    I: SpIndex + Signed,
    Slice: From<Range<I>>,
{
    let (unpad_r, unpad_c) = unpad;
    let mut csx_weights_iter = counts
        .into_iter()
        .zip(signatures.rows())
        .filter(|(csx, _)| csx.nnz() > 0);

    match csx_weights_iter.next() {
        // fastpath if all csx are empty
        None => {
            let shape = (unpad_r.end - unpad_r.start, unpad_c.end - unpad_c.start);
            (
                (Array2::zeros(shape), Array2::zeros(shape)),
                Array2::from_elem(shape, -one::<U>()),
            )
        }
        Some((csx, weights)) => {
            let shape = csx.shape();
            let mut kde = Array2::zeros(shape);

            sparse_kde_csx_(&mut kde, csx.view(), kernel);
            if log {
                kde.mapv_inplace(F::ln_1p);
            }

            let mut kde_norm = kde
                .slice(s![unpad_r.clone(), unpad_c.clone()])
                .map(|k| k.powi(2));
            let mut cosine: Array3<F> = &kde.slice(s![NewAxis, unpad_r.clone(), unpad_c.clone()])
                * &weights.slice(s![.., NewAxis, NewAxis]);

            for (csx, weights) in csx_weights_iter {
                sparse_kde_csx_(&mut kde, csx.view(), kernel);
                let mut kde_unpadded = kde.slice_mut(s![unpad_r.clone(), unpad_c.clone()]);
                if log {
                    kde_unpadded.mapv_inplace(F::ln_1p);
                }

                Zip::from(&mut kde_norm)
                    .and(&kde_unpadded)
                    .for_each(|n, &k| *n += k.powi(2));

                cosine
                    .outer_iter_mut()
                    .zip(&weights)
                    .for_each(|(mut cos, &w)| cos += &kde_unpadded.map(|&x| x * w));
            }
            // TODO: write to zarr
            get_max_cosine_and_celltype(cosine, kde_norm, pairwise_correction)
        }
    }
}

fn get_max_cosine_and_celltype<F, I>(
    cosine: Array3<F>,
    kde_norm: Array2<F>,
    pairwise_correction: ArrayView2<F>,
) -> ((Array2<F>, Array2<F>), Array2<I>)
where
    I: PrimInt + Signed,
    F: NdFloat,
{
    let vars = cosine.map_axis(Axis(0), |view| get_argmax2(view));
    let mut max_cosine = vars.mapv(|(c, _, _, _)| c);
    let mut score = vars.mapv(|(_, s, _, _)| s);
    let mut celltypemap = vars.mapv(|(_, _, i, _)| I::from(i).unwrap());

    Zip::from(&mut celltypemap)
        .and(&mut max_cosine)
        .and(&mut score)
        .and(&vars)
        .and(&kde_norm)
        .for_each(|ct, cos, s, (_, _, i, j), &norm| {
            if norm == zero() {
                *ct = -one::<I>();
            } else {
                let norm_sqrt = norm.sqrt();
                *cos /= norm_sqrt;

                let (i, j) = if i > j { (*i, *j) } else { (*j, *i) };
                *s /= norm_sqrt * pairwise_correction[[i, j]];
            };
        });

    ((max_cosine, score), celltypemap)
}

fn get_argmax2<'a, T: Zero + PartialOrd + Copy + Sub<Output = T>>(
    values: ArrayView1<'a, T>,
) -> (T, T, usize, usize) {
    let mut max = zero();
    let mut max2 = zero();

    let mut argmax = 0;
    let mut argmax2 = 0;

    for (i, &val) in values.indexed_iter() {
        if val > max2 {
            if val > max {
                max2 = max;
                max = val;
                argmax2 = argmax;
                argmax = i;
            } else {
                max2 = val;
                argmax2 = i;
            }
        }
    }
    (max, max - max2, argmax, argmax2)
}

#[cfg(test)]
mod tests {

    use super::*;
    use ndarray::array;

    struct Setup {
        cosine: Array3<f64>,
        norm: Array2<f64>,
        max: Array2<f64>,
        argmax: Array2<i8>,
        cos: Array2<f64>,
        celltype: Array2<i8>,
    }

    impl Setup {
        fn new() -> Self {
            Self {
                cosine: array![[[1.0, 0.0, 0.0]], [[0.5, 1.0, 0.0]]],
                norm: array![[4.0, 1.0, 0.0]],
                max: array![[1.0, 1.0, 0.0]],
                argmax: array![[0, 1, 0]],
                cos: array![[0.5, 1.0, 0.0]],
                celltype: array![[0, 1, -1]],
            }
        }
    }

    // #[test]
    // fn test_max_argmax() {
    //     let setup = Setup::new();

    //     let max_argmax: (Array2<f64>, Array2<i8>) = get_max_argmax(&setup.cosine);

    //     assert_eq!(max_argmax.0, setup.max);
    //     assert_eq!(max_argmax.1, setup.argmax);
    // }

    // #[test]
    // fn test_get_max_cosine_and_celltype() {
    //     let setup = Setup::new();

    //     let cos_ct: ((Array2<f64>, Array2<f64>), Array2<i8>) =
    //         get_max_cosine_and_celltype(setup.cosine, setup.norm);

    //     assert_eq!(cos_ct.0 .0, setup.cos);
    //     assert_eq!(cos_ct.1, setup.celltype);
    // }
}
