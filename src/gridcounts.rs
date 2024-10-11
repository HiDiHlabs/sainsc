use crate::sparsearray_conversion::WrappedCsx;
use crate::utils::create_pool;
use bincode::{deserialize, serialize};
use itertools::Itertools;
use ndarray::Array2;
use num::Zero;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use polars::{
    datatypes::{
        CategoricalOrdering::Physical,
        DataType::{Categorical, Int32, UInt32},
    },
    prelude::*,
};
use polars_arrow::array::{DictionaryArray, UInt32Array, Utf8Array};
use pyo3::{
    exceptions::{PyKeyError, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyType},
};
use pyo3_polars::{error::PyPolarsErr, PyDataFrame};
use rayon::{
    iter::{
        IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
    },
    join, ThreadPool,
};
use sprs::{
    CompressedStorage::{self, CSC, CSR},
    CsMatI, CsMatViewI, SpIndex, TriMatI,
};
use std::{
    cmp::min,
    collections::{HashMap, HashSet},
    iter::repeat,
    ops::AddAssign,
};

/// Class implementation

pub type Count = u32;
pub type CsxIndex = i32;

#[pyclass(mapping, module = "sainsc")]
pub struct GridCounts {
    counts: HashMap<String, CsMatI<Count, CsxIndex>>,
    #[pyo3(get)]
    pub shape: (usize, usize),
    #[pyo3(get)]
    pub resolution: Option<f32>,
    #[pyo3(get)]
    pub n_threads: usize,
    threadpool: ThreadPool,
}

impl GridCounts {
    pub fn get_view(&self, gene: &String) -> Option<CsMatViewI<Count, CsxIndex>> {
        self.counts.get(gene).map(|x| x.view())
    }

    pub fn to_format(&mut self, format: CompressedStorage) {
        self.threadpool.install(|| {
            self.counts.par_iter_mut().for_each(|(_, v)| {
                if format != v.storage() {
                    *v = v.to_other_storage()
                }
            })
        });
    }
}

#[pymethods]
impl GridCounts {
    #[new]
    #[pyo3(signature = (counts, *, resolution=None, n_threads=None))]
    fn new(
        counts: HashMap<String, WrappedCsx<Count, CsxIndex, CsxIndex>>,
        resolution: Option<f32>,
        n_threads: Option<usize>,
    ) -> PyResult<Self> {
        let threadpool =
            create_pool(n_threads).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let counts: HashMap<_, _> =
            threadpool.install(|| counts.into_par_iter().map(|(k, v)| (k, v.0)).collect());

        let shape = if counts.is_empty() {
            (0, 0)
        } else {
            let shapes: Vec<_> = counts.values().map(|v| v.shape()).collect();

            if !shapes.windows(2).all(|w| w[0] == w[1]) {
                return Err(PyValueError::new_err(
                    "All sparse arrays must have same shape",
                ));
            }

            *shapes.first().expect("Length is non-zero")
        };

        let n_threads = threadpool.current_num_threads();

        Ok(Self {
            counts,
            shape,
            resolution,
            n_threads,
            threadpool,
        })
    }

    #[classmethod]
    #[pyo3(signature = (df, *, resolution=None, binsize=None, n_threads=None))]
    fn from_dataframe(
        _cls: &Bound<'_, PyType>,
        df: PyDataFrame,
        resolution: Option<f32>,
        binsize: Option<f32>,
        n_threads: Option<usize>,
    ) -> PyResult<Self> {
        fn _from_dataframe(
            mut df: DataFrame,
            binsize: Option<f32>,
        ) -> Result<
            (
                HashMap<String, CsMatI<Count, CsxIndex, CsxIndex>>,
                (usize, usize),
            ),
            PolarsError,
        > {
            fn col_as_nonull_vec<F, T>(
                df: &DataFrame,
                col: &str,
                f: F,
            ) -> Result<Vec<<T as PolarsNumericType>::Native>, PolarsError>
            where
                F: Fn(&Series) -> Result<&ChunkedArray<T>, PolarsError>,
                T: PolarsNumericType,
            {
                Ok(f(df.column(col)?)?
                    .to_vec_null_aware()
                    .expect_left(&format!("{col} should have no null")))
            }

            // bin if binsize is provided
            if let Some(bin) = binsize {
                df.with_column(df.column("x")? / bin)?;
                df.with_column(df.column("y")? / bin)?;
            }

            // cast to correct dtypes and shift (i.e. subtract min)
            for col in ["x", "y"] {
                let s = df.column(col)?.strict_cast(&Int32)?;
                df.with_column(&s - s.min::<i32>()?.expect("non-null"))?;
            }

            match df.column("count") {
                // if counts does not exist use all 1s
                Err(_) => df.with_column(Series::new("count", vec![1u32; df.height()]))?,
                Ok(s) => df.with_column(s.strict_cast(&UInt32)?)?,
            };

            if !df.column("gene")?.dtype().is_categorical() {
                df.with_column(
                    df.column("gene")?
                        .strict_cast(&Categorical(None, Physical))?,
                )?;
            }

            let shape = (
                df.column("x")?.max::<usize>()?.expect("non-null") + 1,
                df.column("y")?.max::<usize>()?.expect("non-null") + 1,
            );

            let counts_dict = df
                .partition_by(["gene"], true)?
                .into_par_iter()
                .map(|df| {
                    let gene = df
                        .column("gene")?
                        .categorical()?
                        .iter_str()
                        .next()
                        .expect("df must be non-empty")
                        .expect("`gene` must not be null")
                        .to_owned();

                    let x = col_as_nonull_vec(&df, "x", |s| s.i32())?;
                    let y = col_as_nonull_vec(&df, "y", |s| s.i32())?;
                    let counts = col_as_nonull_vec(&df, "count", |s| s.u32())?;

                    Ok::<_, PolarsError>((
                        gene,
                        TriMatI::from_triplets(shape, x, y, counts).to_csr::<CsxIndex>(),
                    ))
                })
                .collect::<Result<_, _>>()?;

            Ok((counts_dict, shape))
        }

        let threadpool =
            create_pool(n_threads).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let n_threads = threadpool.current_num_threads();

        let resolution = resolution.map(|r| r * binsize.unwrap_or(1.));

        match threadpool.install(|| _from_dataframe(df.into(), binsize)) {
            Err(e) => Err(PyPolarsErr::from(e).into()),
            Ok((counts, shape)) => Ok(Self {
                counts,
                shape,
                resolution,
                n_threads,
                threadpool,
            }),
        }
    }

    fn __getitem__(&self, key: String) -> PyResult<WrappedCsx<Count, CsxIndex, CsxIndex>> {
        match self.counts.get(&key) {
            None => Err(PyKeyError::new_err(format!("'{key}' does not exist."))),
            Some(mat) => Ok(WrappedCsx(mat.clone())),
        }
    }

    fn __setitem__(&mut self, key: String, value: WrappedCsx<Count, CsxIndex, CsxIndex>) {
        self.counts.insert(key, value.0);
    }

    fn __delitem__(&mut self, key: String) -> PyResult<()> {
        match self.counts.remove(&key) {
            None => Err(PyKeyError::new_err(key.to_string())),
            Some(_) => Ok(()),
        }
    }

    fn __len__(&self) -> usize {
        self.counts.len()
    }

    fn __contains__(&self, item: String) -> bool {
        self.counts.contains_key(&item)
    }

    fn __eq__(&self, other: &GridCounts) -> bool {
        (self.resolution == other.resolution)
            && (self.shape == other.shape)
            && self.counts.eq(&other.counts)
    }

    fn __ne__(&self, other: &GridCounts) -> bool {
        !self.__eq__(other)
    }

    fn __setstate__(&mut self, state: Bound<'_, PyBytes>) -> PyResult<()> {
        let (counts, shape, resolution, n_threads) = deserialize(state.as_bytes()).unwrap();
        self.counts = counts;
        self.shape = shape;
        self.resolution = resolution;
        self.set_n_threads(n_threads)?;

        Ok(())
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let to_bytes = (&self.counts, self.shape, self.resolution, self.n_threads);

        Ok(PyBytes::new_bound(py, &serialize(&to_bytes).unwrap()))
    }

    fn __getnewargs_ex__(
        &self,
    ) -> PyResult<(
        (HashMap<String, WrappedCsx<Count, CsxIndex, CsxIndex>>,),
        HashMap<String, usize>,
    )> {
        Ok(((HashMap::new(),), HashMap::new()))
    }

    fn __repr__(&self) -> String {
        let mut repr = vec![
            format!("GridCounts ({} threads)", self.n_threads),
            format!("genes: {}", self.counts.len()),
            format!("shape: {:?}", self.shape),
        ];
        if let Some(res) = self.resolution {
            repr.push(format!("resolution: {:.1} nm / px", res));
        }

        repr.join("\n    ")
    }

    fn get(
        &self,
        key: String,
        default: Option<WrappedCsx<Count, CsxIndex, CsxIndex>>,
    ) -> Option<WrappedCsx<Count, CsxIndex, CsxIndex>> {
        match self.__getitem__(key) {
            Ok(x) => Some(x),
            Err(_) => default,
        }
    }

    #[setter]
    fn set_resolution(&mut self, resolution: Option<f32>) -> PyResult<()> {
        match resolution {
            Some(res) if res <= 0. => Err(PyValueError::new_err(
                "`resolution` must be greater than zero",
            )),
            _ => {
                self.resolution = resolution;
                Ok(())
            }
        }
    }

    #[setter]
    fn set_n_threads(&mut self, n_threads: Option<usize>) -> PyResult<()> {
        self.threadpool =
            create_pool(n_threads).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.n_threads = self.threadpool.current_num_threads();

        Ok(())
    }

    fn genes(&self) -> Vec<String> {
        self.counts.keys().cloned().collect()
    }

    fn gene_counts(&self) -> HashMap<String, Count> {
        self.threadpool.install(|| {
            self.counts
                .par_iter()
                .map(|(gene, mat)| (gene.to_owned(), mat.data().iter().sum()))
                .collect()
        })
    }

    fn grid_counts(&self) -> Py<PyArray2<Count>> {
        fn triplet_to_dense<N: Copy + Zero + AddAssign<N>, I: SpIndex>(
            coo: TriMatI<N, I>,
        ) -> Array2<N> {
            let mut dense: Array2<N> = Array2::zeros(coo.shape());
            coo.triplet_iter().for_each(|(v, (i, j))| {
                dense[[
                    i.to_usize().expect("valid index"),
                    j.to_usize().expect("valid index"),
                ]] += *v;
            });

            dense
        }
        let (v, (i, j)) = self.threadpool.install(|| {
            let (v, (i, j)): (Vec<_>, (Vec<_>, Vec<_>)) = self
                .counts
                .par_iter()
                .map(|(_, mat)| -> (Vec<_>, (Vec<_>, Vec<_>)) { mat.iter().multiunzip() })
                .unzip();

            join(|| v.concat(), || join(|| i.concat(), || j.concat()))
        });

        let gridcounts = triplet_to_dense(TriMatI::from_triplets(self.shape, i, j, v));
        Python::with_gil(|py| gridcounts.into_pyarray_bound(py).unbind())
    }

    fn select_genes(&mut self, genes: HashSet<String>) {
        self.counts.retain(|k, _| genes.contains(k));
    }

    #[pyo3(signature = (min=1, max=Count::MAX))]
    fn filter_genes_by_count(&mut self, min: Count, max: Count) {
        let genes: HashSet<_> = self.threadpool.install(|| {
            self.gene_counts()
                .into_par_iter()
                .filter_map(|(gene, count)| {
                    if (count >= min) & (count <= max) {
                        Some(gene)
                    } else {
                        None
                    }
                })
                .collect()
        });
        self.select_genes(genes);
    }

    fn crop(
        &mut self,
        x: (Option<usize>, Option<usize>),
        y: (Option<usize>, Option<usize>),
    ) -> PyResult<()> {
        let x_start = x.0.unwrap_or(0);
        let y_start = y.0.unwrap_or(0);
        let x_end = x.1.map_or(self.shape.0, |x| min(x, self.shape.0));
        let y_end = y.1.map_or(self.shape.1, |x| min(x, self.shape.1));

        if (x_end <= x_start) || (y_end <= y_start) {
            return Err(PyValueError::new_err("Trying to crop with empty slice."));
        }

        self.threadpool.install(|| {
            self.counts.par_iter_mut().for_each(|(_, mat)| {
                let (outer, inner) = match mat.storage() {
                    CSR => (x_start..x_end, y_start..y_end),
                    CSC => (y_start..y_end, x_start..x_end),
                };
                *mat = mat
                    .slice_outer(outer)
                    .transpose_into()
                    .to_other_storage()
                    .slice_outer(inner)
                    .transpose_into()
                    .to_owned();
            });
        });

        self.shape = (x_end - x_start, y_end - y_start);
        Ok(())
    }

    fn filter_mask(&mut self, mask: PyReadonlyArray2<'_, bool>) {
        let mask = mask.as_array();

        self.threadpool.install(|| {
            self.counts.par_iter_mut().for_each(|(_, mat)| {
                let (data, x, y) = mat
                    .into_iter()
                    .filter(|(_, (i, j))| mask[[*i as usize, *j as usize]])
                    .map(|(v, (i, j))| (v, i, j))
                    .multiunzip();

                *mat = TriMatI::from_triplets(self.shape, x, y, data).to_csr();
            });
        });
    }

    fn as_dataframe(&mut self) -> PyResult<PyDataFrame> {
        self.to_format(CSR);

        let genes: Vec<_> = self.counts.keys().sorted().collect();

        let ((counts, (x, y)), gene_idx): ((Vec<&Count>, (Vec<_>, Vec<_>)), Vec<_>) = genes
            .iter()
            .zip(0u32..)
            .flat_map(|(&gene, i)| {
                self.get_view(gene)
                    .expect("gene exists because we collected the keys above")
                    .iter_rbr()
                    .zip(repeat(i))
            })
            .multiunzip();

        let counts = Series::from_iter(counts).with_name("count");
        let x = Series::from_vec("x", x);
        let y = Series::from_vec("y", y);
        // construct categorical gene array from codes and categories
        let genes = Series::from_arrow(
            "gene",
            Box::new(
                DictionaryArray::try_from_keys(
                    UInt32Array::from_vec(gene_idx),
                    Box::new(Utf8Array::<i32>::from_iter(genes.into_iter().map(Some))),
                )
                .map_err(PyPolarsErr::from)?,
            ),
        )
        .map_err(PyPolarsErr::from)?;

        let df = DataFrame::new(vec![genes, x, y, counts]).map_err(PyPolarsErr::from)?;

        Ok(PyDataFrame(df))
    }
}
