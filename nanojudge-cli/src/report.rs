/// Sink for diagnostic, warning, and progress output (everything the CLI
/// normally writes to stderr).
pub trait Reporter: Send + Sync {
    /// Emit one line of output (a terminal sink appends a newline).
    fn line(&self, msg: &str);

    /// Emit raw text without a trailing newline (used for inline progress dots).
    fn raw(&self, text: &str);
}

/// Default CLI reporter: writes lines via `eprintln!` and raw text via `eprint!`.
#[derive(Debug, Default, Clone, Copy)]
pub struct StderrReporter;

impl Reporter for StderrReporter {
    fn line(&self, msg: &str) {
        eprintln!("{msg}");
    }

    fn raw(&self, text: &str) {
        eprint!("{text}");
    }
}

/// No-op reporter for quiet library calls and tests.
#[derive(Debug, Default, Clone, Copy)]
pub struct SilentReporter;

impl Reporter for SilentReporter {
    fn line(&self, _msg: &str) {}
    fn raw(&self, _text: &str) {}
}
