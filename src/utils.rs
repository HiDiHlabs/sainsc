use rayon::{ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder};

/// Create rayon ThreadPool with n threads
pub fn create_pool(n_threads: usize) -> Result<ThreadPool, ThreadPoolBuildError> {
    ThreadPoolBuilder::new().num_threads(n_threads).build()
}
