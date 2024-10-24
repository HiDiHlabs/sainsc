use ndarray::{Array2, Array3, NdFloat};
use num::zero;
use serde_json::Map;
use std::{collections::BTreeMap, error::Error, path::PathBuf, sync::Arc};
use zarrs::{
    array::{codec::GzipCodec, Array as ZarrArray, ArrayBuilder, DataType, Element, FillValue},
    group::{Group, GroupBuilder, GroupMetadataV3},
    storage::store::FilesystemStore,
};

const CT_PATH_PREFIX: &str = "/cosine";
const KDE_PATH: &str = "/kde";

pub struct ZarrChunkInfo {
    pub store: Arc<FilesystemStore>,
    pub celltypes: Vec<String>, // TODO: improve
    pub chunk_idx: Vec<u64>,
}

pub fn initialize_cosine_zarrstore(
    path: PathBuf,
    celltypes: &[String],
    shape: (usize, usize),
    chunk_size: (usize, usize),
) -> Result<Arc<FilesystemStore>, Box<dyn Error + Send + Sync>> {
    let store = Arc::new(FilesystemStore::new(path)?);

    let shape = vec![shape.0 as u64, shape.1 as u64];
    let chunk_shape = vec![chunk_size.0 as u64, chunk_size.1 as u64];

    // generate root
    let mut root_meta = Map::new();
    root_meta.insert("shape".into(), serde_json::to_value(shape.clone())?);
    root_meta.insert("celltypes".into(), serde_json::to_value(celltypes)?);
    root_meta.insert("chunk_size".into(), serde_json::to_value(chunk_size)?);

    Group::new_with_metadata(
        store.clone(),
        "/",
        GroupMetadataV3::new(root_meta, BTreeMap::new()).into(),
    )?
    .store_metadata()?;

    // generate cosine group
    GroupBuilder::new()
        .build(store.clone(), CT_PATH_PREFIX)?
        .store_metadata()?;

    let mut array_builder = ArrayBuilder::new(
        shape,
        DataType::Float32,
        chunk_shape.try_into()?,
        FillValue::from(0),
    );
    array_builder
        .bytes_to_bytes_codecs(vec![Box::new(GzipCodec::new(5)?)])
        .dimension_names(["x", "y"].into());

    // generate empty arrays for celltypes
    for ct in celltypes {
        let ct_array = array_builder.build(store.clone(), &format!("{CT_PATH_PREFIX}/{ct}"))?;
        ct_array.store_metadata()?;
    }

    // generate empty array for kde
    let kde_array = array_builder.build(store.clone(), KDE_PATH)?;
    kde_array.store_metadata()?;

    Ok(store)
}

pub fn write_cosine_to_zarr<T: NdFloat + Element>(
    zarr_store: Arc<FilesystemStore>,
    cosine: &Array3<T>,
    kde_norm: &Array2<T>,
    celltypes: &[String],
    chunk_idx: &[u64],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // let ct_group = Group::open(zarr_store.clone(), CT_PATH_PREFIX)?;

    for (cos, ct) in cosine.outer_iter().zip(celltypes) {
        let mut cos_norm = &cos / kde_norm;
        cos_norm.mapv_inplace(|v| if v.is_nan() { zero() } else { v });

        ZarrArray::open(zarr_store.clone(), &format!("{CT_PATH_PREFIX}/{ct}"))?
            .store_chunk_ndarray(chunk_idx, cos_norm)?;
    }

    ZarrArray::open(zarr_store.clone(), KDE_PATH)?
        .store_chunk_ndarray(chunk_idx, kde_norm.clone())?;

    Ok(())
}
