use std::fmt;

/// Coarse error categories for nanojudge-cli.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Invalid config file, CLI flags, or judge settings.
    Config(String),
    /// Invalid items, criteria, prompt template, or loaded judgements file.
    Input(String),
    /// Endpoint probe rejected one or more judges.
    Probe(String),
    /// Fatal mid-run protocol violation from an endpoint (e.g. missing logprobs,
    /// inconsistent token counts, conflicting reasoning fields).
    Endpoint(String),
    /// Run finished with no judgements or edges to score.
    NoResults(String),
    /// Filesystem read/write or OS directory/clock error.
    Io(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config(msg)
            | Error::Input(msg)
            | Error::Probe(msg)
            | Error::Endpoint(msg)
            | Error::NoResults(msg)
            | Error::Io(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
