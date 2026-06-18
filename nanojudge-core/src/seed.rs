use rand::rngs::StdRng;
use rand::SeedableRng;

pub const SUBSYSTEM_PAIRING: u64 = 1;
pub const SUBSYSTEM_MCMC: u64 = 2;
pub const SUBSYSTEM_JUDGE_ASSIGN: u64 = 3;
pub const SUBSYSTEM_TEMP_JITTER: u64 = 4;
pub const SUBSYSTEM_BENCHMARK: u64 = 5;

pub fn make_rng(seed: Option<u64>, subsystem: u64) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s.wrapping_add(subsystem)),
        None => StdRng::from_os_rng(),
    }
}
