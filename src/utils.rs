use rayon::{ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder};

/// Create rayon ThreadPool with n threads
pub fn create_pool(n_threads: Option<usize>) -> Result<ThreadPool, ThreadPoolBuildError> {
    ThreadPoolBuilder::new()
        .num_threads(n_threads.unwrap_or(0))
        .build()
}
