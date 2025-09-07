use std::time::Duration;

use claudius::Usage as ClaudiusUsage;

/// Usage metrics for PolicyAI operations.
///
/// This tracks both the token usage from claudius and additional metrics
/// like wall clock time and iteration count for policy application.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Usage {
    /// Total token usage across all API calls
    pub claudius_usage: Option<ClaudiusUsage>,
    /// Wall clock time for the operation
    pub wall_clock_time: Duration,
    /// Number of iterations needed (for retry logic)
    pub iterations: usize,
}

impl Usage {
    /// Create a new empty Usage
    pub fn new() -> Self {
        Self::default()
    }

    /// Add claudius usage to the total
    pub fn add_claudius_usage(&mut self, usage: ClaudiusUsage) {
        self.claudius_usage = Some(match self.claudius_usage {
            Some(existing) => existing + usage,
            None => usage,
        });
    }

    /// Increment the iteration counter
    pub fn increment_iterations(&mut self) {
        self.iterations += 1;
    }

    /// Set the wall clock time
    pub fn set_wall_clock_time(&mut self, duration: Duration) {
        self.wall_clock_time = duration;
    }
}
