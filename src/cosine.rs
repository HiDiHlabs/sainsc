use crate::gridcounts::{GridCounts, GridCountsView};
use crate::sparsekde::sparse_kde_csx_;
use crate::utils::create_pool;

use itertools::Itertools;
use ndarray::{
    concatenate, s, Array2, Array3, ArrayView1, ArrayView2, Axis, NdFloat, NewAxis, ShapeError,
    Slice, Zip,
};
use num::{one, zero, NumCast, PrimInt, Signed};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use sprs::{CsMatI, SpIndex};
use std::{error::Error, iter::Sum, ops::Range};

macro_rules! build_cos_ct_fn {
    ($name:tt, $t_cos:ty, $t_ct:ty) => {
        #[pyfunction]
        #[pyo3(signature = (counts, genes, signatures, kernel, *, log=false, min_transcripts=None, chunk_size=(500, 500), n_threads=None))]
        /// calculate cosine similarity and assign celltype
        pub fn $name<'py>(
            py: Python<'py>,
            counts: &mut GridCounts,
            genes: Vec<String>,
            signatures: PyReadonlyArray2<'py, $t_cos>,
            kernel: PyReadonlyArray2<'py, $t_cos>,
            log: bool,
            min_transcripts: Option<u32>,
            chunk_size: (usize, usize),
            n_threads: Option<usize>,
        ) -> PyResult<(
            Bound<'py, PyArray2<$t_cos>>,
            Bound<'py, PyArray2<$t_cos>>,
            Bound<'py, PyArray2<$t_ct>>,
        )> {

            let gene_counts = counts.get_views(genes).ok_or(PyValueError::new_err("Not all genes exist"))?;

            let cos_ct = chunk_and_calculate_cosine(
                gene_counts,
                signatures.as_array(),
                kernel.as_array(),
                log,
                min_transcripts,
                chunk_size,
                n_threads
            );

            match cos_ct {
                Ok((cosine, score, celltype_map)) => Ok((
                    cosine.into_pyarray(py),
                    score.into_pyarray(py),
                    celltype_map.into_pyarray(py),
                )),
                Err(e) => Err(PyValueError::new_err(e.to_string())),
            }
        }
    };
}

build_cos_ct_fn!(cosinef32_and_celltypei8, f32, i8);
build_cos_ct_fn!(cosinef32_and_celltypei16, f32, i16);

fn chunk_and_calculate_cosine<C, I, F, U>(
    counts: GridCountsView<C, I>,
    signatures: ArrayView2<F>,
    kernel: ArrayView2<F>,
    log: bool,
    min_transcripts: Option<C>,
    chunk_size: (usize, usize),
    n_threads: Option<usize>,
) -> Result<(Array2<F>, Array2<F>, Array2<U>), Box<dyn Error>>
where
    C: NumCast + Copy + Sync + Send + Default + PartialOrd + Sum + for<'a> Sum<&'a C>,
    I: SpIndex + Signed + Sync + Send,
    F: NdFloat,
    U: PrimInt + Signed + Sync + Send,
    Slice: From<Range<I>>,
{
    let pool = create_pool(n_threads)?;

    let pad = get_padding(kernel.shape());
    let (m, n) = n_chunks(counts.shape, chunk_size); // number of chunks

    let signature_similarity_correction = similarity_correction(&signatures);

    let ((cosine, score), celltype): ((Vec<_>, Vec<_>), Vec<_>) = pool.install(|| {
        // generate all chunk indices
        let chunk_indices: Vec<_> = (0..m).cartesian_product(0..n).collect();

        // chunk and calculate cosine/celltype in parallel
        chunk_indices
            .into_par_iter()
            .map(|idx| {
                let (chunk, unpad) = counts.get_chunk(idx, chunk_size, pad);

                cosine_and_celltype_(
                    chunk,
                    signatures,
                    &signature_similarity_correction,
                    kernel,
                    unpad,
                    log,
                    min_transcripts,
                )
            })
            .unzip()
    });

    // concatenate all chunks back to original shape
    Ok((
        concat_2d(&cosine, n)?,
        concat_2d(&score, n)?,
        concat_2d(&celltype, n)?,
    ))
}

fn n_chunks(shape: (usize, usize), chunk_shape: (usize, usize)) -> (usize, usize) {
    let (nrow, ncol) = shape;
    let (srow, scol) = chunk_shape;
    (nrow.div_ceil(srow), ncol.div_ceil(scol))
}

fn get_padding(shape: &[usize]) -> (usize, usize) {
    ((shape[0] - 1) / 2, (shape[1] - 1) / 2)
}

fn similarity_correction<T: NdFloat>(arr: &ArrayView2<T>) -> Array2<T> {
    let n_cols = arr.ncols();
    Array2::from_shape_fn((n_cols, n_cols), |(i, j)| {
        if i != j {
            let sig1 = arr.index_axis(Axis(1), i);
            let sig2 = arr.index_axis(Axis(1), j);
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
    })
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

fn cosine_and_celltype_<C, I, F, U>(
    counts: Vec<CsMatI<C, I>>,
    signatures: ArrayView2<F>,
    pairwise_correction: &Array2<F>,
    kernel: ArrayView2<F>,
    unpad: (Range<usize>, Range<usize>),
    log: bool,
    min_transcripts: Option<C>,
) -> ((Array2<F>, Array2<F>), Array2<U>)
where
    C: NumCast + Copy + PartialOrd + Sum + for<'a> Sum<&'a C>,
    F: NdFloat,
    U: PrimInt + Signed,
    I: SpIndex + Signed,
    Slice: From<Range<I>>,
{
    let mut sufficient_transcripts = true;
    if let Some(min_t) = min_transcripts {
        let n_transcripts: C = counts
            .iter()
            .map(|gene| gene.data().iter().sum::<C>())
            .sum();
        sufficient_transcripts = n_transcripts >= min_t;
    };

    let mut csx_weights_iter = counts
        .into_iter()
        .zip(signatures.rows())
        .filter(|(csx, _)| csx.nnz() > 0);

    match csx_weights_iter.next() {
        Some((csx, weights)) if sufficient_transcripts => {
            let shape = csx.shape();
            let mut kde = Array2::zeros(shape);
            let kde_slice = s![unpad.0, unpad.1];

            sparse_kde_csx_(&mut kde, &csx, kernel);

            let mut kde_unpadded = kde.slice_mut(kde_slice);
            if log {
                kde_unpadded.mapv_inplace(F::ln_1p);
            }

            let mut kde_norm = kde_unpadded.map(|k| k.powi(2));
            let mut cosine: Array3<F> =
                &kde_unpadded.slice(s![NewAxis, .., ..]) * &weights.slice(s![.., NewAxis, NewAxis]);

            for (csx, weights) in csx_weights_iter {
                sparse_kde_csx_(&mut kde, &csx, kernel);
                let mut kde_unpadded = kde.slice_mut(kde_slice);
                if log {
                    kde_unpadded.mapv_inplace(F::ln_1p);
                }

                Zip::from(&mut kde_norm)
                    .and(&kde_unpadded)
                    .for_each(|n, &k| *n += k.powi(2));

                cosine
                    .outer_iter_mut()
                    .zip(&weights)
                    .filter(|(_, &w)| w != zero::<F>())
                    .for_each(|(mut cos, &w)| cos += &kde_unpadded.map(|&x| x * w));
            }
            kde_norm.mapv_inplace(F::sqrt);
            // TODO: write to zarr
            get_max_cosine_and_celltype(cosine, kde_norm, pairwise_correction)
        }
        // fastpath if all csx are empty or too few transcripts
        _ => {
            let shape = (unpad.0.end - unpad.0.start, unpad.1.end - unpad.1.start);
            (
                (
                    Array2::from_elem(shape, F::nan()),
                    Array2::from_elem(shape, F::nan()),
                ),
                Array2::from_elem(shape, -one::<U>()),
            )
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
    let celltypemap = vars.mapv(|(_, _, i)| i);

    max_cosine /= &kde_norm;
    score /= &kde_norm;

    ((max_cosine, score), celltypemap)
}

fn get_argmax2<T: NdFloat, I: Signed + PrimInt>(
    values: ArrayView1<T>,
    pairwise_correction: &Array2<T>,
) -> (T, T, I) {
    let mut max = zero();
    let mut max2 = zero();

    let mut argmax = -one::<I>();
    let mut argmax2 = -one::<I>();

    for (i, &val) in values.indexed_iter() {
        if val > max2 {
            if val > max {
                max2 = max;
                max = val;
                argmax2 = argmax;
                argmax = I::from(i).expect("correct type must be selected beforehand");
            } else {
                max2 = val;
                argmax2 = I::from(i).expect("correct type must be selected beforehand");
            }
        }
    }
    let score = if (argmax >= zero()) & (argmax2 >= zero()) {
        let i = argmax.to_usize().expect("non-negative");
        let j = argmax2.to_usize().expect("non-negative");

        (max - max2) / pairwise_correction[[i, j]]
    } else {
        // TODO: what to return if only one signature is non-zero, what would the c
        // orrection-factor be, the max of the row?
        T::nan()
    };
    (max, score, argmax)
}

// #[cfg(test)]
// mod tests {

//     use super::*;
//     use ndarray::array;

//     struct Setup {
//         cosine: Array3<f64>,
//         norm: Array2<f64>,
//         max: Array2<f64>,
//         argmax: Array2<i8>,
//         cos: Array2<f64>,
//         celltype: Array2<i8>,
//     }

//     impl Setup {
//         fn new() -> Self {
//             Self {
//                 cosine: array![[[1.0, 0.0, 0.0]], [[0.5, 1.0, 0.0]]],
//                 norm: array![[4.0, 1.0, 0.0]],
//                 max: array![[1.0, 1.0, 0.0]],
//                 argmax: array![[0, 1, 0]],
//                 cos: array![[0.5, 1.0, 0.0]],
//                 celltype: array![[0, 1, -1]],
//             }
//         }
//     }

//     #[test]
//     fn test_max_argmax() {
//         let setup = Setup::new();

//         let max_argmax: (Array2<f64>, Array2<i8>) = get_max_argmax(&setup.cosine);

//         assert_eq!(max_argmax.0, setup.max);
//         assert_eq!(max_argmax.1, setup.argmax);
//     }

//     #[test]
//     fn test_get_max_cosine_and_celltype() {
//         let setup = Setup::new();

//         let cos_ct: ((Array2<f64>, Array2<f64>), Array2<i8>) =
//             get_max_cosine_and_celltype(setup.cosine, setup.norm);

//         assert_eq!(cos_ct.0 .0, setup.cos);
//         assert_eq!(cos_ct.1, setup.celltype);
//     }
// }
