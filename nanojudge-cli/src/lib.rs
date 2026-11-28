pub mod args;
pub mod benchmark;
pub mod config;
pub mod error;
pub mod items;
pub mod llm;
pub mod output;
pub mod parse;
pub mod probe;
pub mod prompt;
pub mod rank;
pub mod resolve;
pub mod score;

pub use benchmark::{BenchmarkReport, LatencyStats};
pub use error::{Error, Result};
pub use output::RankingOutput;
pub use probe::JudgeProbeResult;

pub(crate) const DEFAULT_BENCHMARK_PAIRS: &str = "100";

pub(crate) const DEFAULT_CONFIDENCE_LEVEL: f64 = 0.95;
pub(crate) const DEFAULT_REGULARIZATION_STRENGTH: f64 = 0.01;
pub(crate) const DEFAULT_PRIOR_TAU2: f64 = 10.0;
pub(crate) const DEFAULT_BIAS_PRIOR_TAU2: f64 = 2.0;
pub(crate) const DEFAULT_BIAS_PRIOR: f64 = 0.5;
