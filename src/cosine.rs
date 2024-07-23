use crate::gridcounts::GridCounts;
use crate::sparsekde::sparse_kde_csx_;
use crate::utils::create_pool;

use ndarray::{
    concatenate, s, Array2, Array3, ArrayView1, ArrayView2, Axis, NdFloat, NewAxis, ShapeError,
    Slice, Zip,
};
use num::{one, zero, NumCast, PrimInt, Signed};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use sprs::{CompressedStorage::CSR, CsMatI, CsMatViewI, SpIndex};
use std::{
    cmp::{max, min},
    error::Error,
    ops::Range,
};

macro_rules! build_cos_ct_fn {
    ($name:tt, $t_cos:ty, $t_ct:ty) => {
        #[pyfunction]
        #[pyo3(signature = (counts, genes, signatures, kernel, *, log=false, low_memory=true, chunk_size=(500, 500), n_threads=None))]
        /// calculate cosine similarity and assign celltype
        pub fn $name<'py>(
            py: Python<'py>,
            counts: &mut GridCounts,
            genes: Vec<String>,
            signatures: PyReadonlyArray2<'py, $t_cos>,
            kernel: PyReadonlyArray2<'py, $t_cos>,
            log: bool,
            low_memory: bool,
            chunk_size: (usize, usize),
            n_threads: Option<usize>,
        ) -> PyResult<(
            Bound<'py, PyArray2<$t_cos>>,
            Bound<'py, PyArray2<$t_cos>>,
            Bound<'py, PyArray2<$t_ct>>,
        )> {
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
                low_memory,
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

fn chunk_and_calculate_cosine<C, I, F, U>(
    counts: &[CsMatViewI<C, I>],
    signatures: ArrayView2<F>,
    kernel: ArrayView2<F>,
    shape: (usize, usize),
    log: bool,
    low_memory: bool,
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
            if i != j {
                let sig1 = signatures.index_axis(Axis(1), i);
                let sig2 = signatures.index_axis(Axis(1), j);
                // technically we want the dot_product of s=(sig1-sig2) with a vector where
                // the negative dimensions of this vector are set to zero (x),
                // but these will then cancel out anyway so we can simplify to using the
                // dot product with itself s . x => x . x
                // additional we need to divide by the norm of x
                // as the norm is the sqrt of the dot product with itself (which we
                // already calculated) divided by its sqrt we end up with
                // s . x / norm(x) = x . x / sqrt(x . x) = sqrt(x . x)
                let x = (&sig1 - &sig2).mapv(|x| if x <= zero() { zero() } else { x });
                x.dot(&x).sqrt()
            } else {
                zero()
            }
        });

    let (cosine, score, celltype) = pool.install(|| {
        if low_memory {
            let mut cosine = Vec::with_capacity(m * n);
            let mut score = Vec::with_capacity(m * n);
            let mut celltype = Vec::with_capacity(m * n);

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
                            &signature_similarity_correction,
                            kernel,
                            (unpad_row.clone(), unpad_col),
                            log,
                        )
                    })
                    .unzip();
                cosine.extend(cosine_cols);
                score.extend(score_cols);
                celltype.extend(celltype_cols);
            }
            (cosine, score, celltype)
        } else {
            let mut chunk_info = Vec::with_capacity(m * n);
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
                for j in 0..n {
                    let (slice_col, unpad_col) = chunk_(j, scol, ncol, padcol);
                    let chunk: Vec<_> = row_chunk
                        .par_iter()
                        .map(|c| c.slice_outer(slice_col.clone()).transpose_into().to_owned())
                        .collect();
                    chunk_info.push(((unpad_row.clone(), unpad_col), chunk));
                }
            }

            let ((cosine, score), celltype): ((Vec<Array2<F>>, Vec<Array2<F>>), Vec<Array2<U>>) =
                chunk_info
                    .into_par_iter()
                    .map(|(unpad, chunk)| {
                        cosine_and_celltype_(
                            chunk,
                            signatures,
                            &signature_similarity_correction,
                            kernel,
                            unpad,
                            log,
                        )
                    })
                    .unzip();
            (cosine, score, celltype)
        }
    });

    // concatenate all chunks back to original shape
    let cosine = concat_2d(&cosine, n)?;
    let score = concat_2d(&score, n)?;
    let celltype = concat_2d(&celltype, n)?;

    Ok((cosine, score, celltype))
}

fn concat_1d<T: Clone + Sync + Send>(
    chunks: &[Array2<T>],
    axis: usize,
) -> Result<Array2<T>, ShapeError> {
    concatenate(
        Axis(axis),
        &chunks.par_iter().map(|a| a.view()).collect::<Vec<_>>(),
    )
}

fn concat_2d<T: Clone + Sync + Send>(
    chunks: &[Array2<T>],
    size: usize,
) -> Result<Array2<T>, ShapeError> {
    concat_1d(
        &(chunks
            .chunks(size)
            .map(|col| concat_1d(col, 1))
            .collect::<Result<Vec<_>, _>>()?),
        0,
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

fn cosine_and_celltype_<C, I, F, U>(
    counts: Vec<CsMatI<C, I>>,
    signatures: ArrayView2<F>,
    pairwise_correction: &Array2<F>,
    kernel: ArrayView2<F>,
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
    pairwise_correction: &Array2<F>,
) -> ((Array2<F>, Array2<F>), Array2<I>)
where
    I: PrimInt + Signed,
    F: NdFloat,
{
    let vars = cosine.map_axis(Axis(0), |view| get_argmax2(view, pairwise_correction));
    let mut max_cosine = vars.mapv(|(c, _, _)| c);
    let mut score = vars.mapv(|(_, s, _)| s);
    let mut celltypemap = vars.mapv(|(_, _, i)| I::from(i).unwrap());

    Zip::from(&mut celltypemap)
        .and(&mut max_cosine)
        .and(&mut score)
        .and(&kde_norm)
        .for_each(|ct, cos, s, &norm| {
            if norm == zero() {
                *ct = -one::<I>();
                *s = zero();
            } else {
                let norm_sqrt = norm.sqrt();
                *cos /= norm_sqrt;
                *s /= norm_sqrt;
            };
        });

    ((max_cosine, score), celltypemap)
}

fn get_argmax2<T: NdFloat>(
    values: ArrayView1<T>,
    pairwise_correction: &Array2<T>,
) -> (T, T, usize) {
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
    let score = (max - max2) / pairwise_correction[[argmax, argmax2]];
    (max, score, argmax)
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
