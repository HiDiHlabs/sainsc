use crate::gridcounts::{GridCounts, GridFloats};
use crate::utils::create_pool;
use num::zero;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use sprs::TriMatI;
use std::collections::HashMap;

#[pyfunction]
#[pyo3(signature = (counts, scale, bin_ratio, *, n_threads=None))]
/// Correct ambient RNA from cellbender output
pub fn correct_ambient_rna_rs(
    counts: &GridCounts,
    scale: &GridFloats,
    bin_ratio: usize,
    n_threads: Option<usize>,
) -> PyResult<GridFloats> {
    let n_threads = n_threads.unwrap_or(counts.n_threads);
    let threadpool = create_pool(n_threads).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let shape = counts.shape;

    let corrected_counts: HashMap<_, _> = threadpool.install(|| {
        scale
            .counts
            .par_iter()
            .map(|(gene, scale)| {
                let ((rows, cols), values): ((Vec<_>, Vec<_>), Vec<_>) = counts
                    .get_view(gene)
                    .expect("Gene must exist in `counts` and `scale`")
                    .iter()
                    .map(|(&v, (i, j))| {
                        let i2 = i as usize / bin_ratio;
                        let j2 = j as usize / bin_ratio;
                        let v_scaled = v as f32 * scale.get(i2, j2).unwrap_or(&zero());
                        ((i, j), v_scaled)
                    })
                    .unzip();
                (
                    gene.clone(),
                    TriMatI::from_triplets(shape, rows, cols, values).to_csr(),
                )
            })
            .collect()
    });

    GridFloats::new(corrected_counts, counts.resolution, Some(n_threads))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}
