use guacamole::{combinators, Guacamole};

/// Deterministic stream of UUID-shaped mask names.
#[derive(Clone)]
pub(crate) struct MaskNameGenerator {
    guac: Guacamole,
}

impl MaskNameGenerator {
    /// Create a generator at the beginning of the guacamole seed-0 stream.
    pub(crate) fn new() -> Self {
        Self {
            guac: Guacamole::new(0),
        }
    }

    /// Return the next UUID-shaped value from the deterministic stream.
    pub(crate) fn next(&mut self) -> String {
        combinators::uuid(&mut self.guac)
    }
}

impl Default for MaskNameGenerator {
    fn default() -> Self {
        Self::new()
    }
}
