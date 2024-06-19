use numpy::{Element, IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    sync::GILOnceCell,
};
use sprs::{
    CompressedStorage::{CSC, CSR},
    CsMatBase, CsMatI, SpIndex,
};

// cache scipy imports
static SP_SPARSE: GILOnceCell<Py<PyModule>> = GILOnceCell::new();
static CSR_ARRAY: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
static CSC_ARRAY: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
static SPARRAY: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
static SPMATRIX: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

// implement WrappedCsxView?

/// Conversion type for sprs::CsMat <-> scipy.sparse.csx_array
pub struct WrappedCsx<N, I: SpIndex, Iptr: SpIndex>(pub CsMatI<N, I, Iptr>);

fn get_scipy_sparse(py: Python) -> PyResult<&Py<PyModule>> {
    SP_SPARSE.get_or_try_init(py, || Ok(py.import_bound("scipy.sparse")?.unbind()))
}

fn get_scipy_sparse_attr(py: Python, attr: &str) -> PyResult<PyObject> {
    get_scipy_sparse(py)?.getattr(py, attr)
}

/// Return a CsMat in SciPy CSX tuple order
pub fn make_csx_tuple<D, I, Iptr>(
    py: Python<'_>,
    cs: CsMatI<D, I, Iptr>,
) -> (
    Bound<'_, PyArray1<D>>,
    Bound<'_, PyArray1<I>>,
    Bound<'_, PyArray1<Iptr>>,
)
where
    D: Element,
    I: Element + SpIndex,
    Iptr: Element + SpIndex,
{
    let (indptr, indices, data) = cs.into_raw_storage();

    return (
        data.into_pyarray_bound(py),
        indices.into_pyarray_bound(py),
        indptr.into_pyarray_bound(py),
    );
}

impl<N: Element, I: SpIndex + Element, Iptr: SpIndex + Element> IntoPy<PyObject>
    for WrappedCsx<N, I, Iptr>
{
    fn into_py(self, py: Python<'_>) -> PyObject {
        let csx = self.0;
        let shape = csx.shape();

        let sparray = match csx.storage() {
            CSR => CSR_ARRAY.get_or_try_init(py, || get_scipy_sparse_attr(py, "csr_array")),
            CSC => CSC_ARRAY.get_or_try_init(py, || get_scipy_sparse_attr(py, "csc_array")),
        };

        sparray
            .unwrap()
            .call1(py, (make_csx_tuple(py, csx), shape))
            .unwrap()
            .extract(py)
            .unwrap()
    }
}
impl<'py, N: Element, I: SpIndex + Element, Iptr: SpIndex + Element> FromPyObject<'py>
    for WrappedCsx<N, I, Iptr>
{
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        fn boundpyarray_to_vec<T: Element>(obj: Bound<'_, PyAny>) -> PyResult<Vec<T>> {
            Ok(obj.extract::<PyReadonlyArray1<T>>()?.as_array().to_vec())
        }

        Python::with_gil(|py| {
            let sparray = SPARRAY
                .get_or_try_init(py, || get_scipy_sparse_attr(py, "sparray"))?
                .bind(py);
            let spmatrix = SPMATRIX
                .get_or_try_init(py, || get_scipy_sparse_attr(py, "spmatrix"))?
                .bind(py);

            let format = obj.getattr("format")?;
            if !((obj.is_instance(spmatrix)? || obj.is_instance(sparray)?)
                && ((format.eq("csr")?) || (format.eq("csc")?)))
            {
                Err(PyTypeError::new_err(
                    "Only `sparray`/`spmatrix` with format 'csr' or 'csc' can be extracted.",
                ))
            } else {
                let shape = obj.getattr("shape")?.extract()?;

                let data = boundpyarray_to_vec(obj.getattr("data")?)?;
                let indices = boundpyarray_to_vec(obj.getattr("indices")?)?;
                let indptr = boundpyarray_to_vec(obj.getattr("indptr")?)?;

                let csx = if format.eq("csr")? {
                    CsMatBase::new_from_unsorted(shape, indptr, indices, data)
                } else {
                    CsMatBase::new_from_unsorted_csc(shape, indptr, indices, data)
                };
                match csx {
                    Ok(csx) => Ok(WrappedCsx(csx)),
                    Err((.., e)) => Err(PyValueError::new_err(e.to_string())),
                }
            }
        })
    }
}
