//! Mask generation for creating values unlikely to be in LLM training data.
//!
//! This module provides functionality to generate pseudorandom masks that can be used
//! to replace values with strings that are unlikely to appear in an LLM's training data,
//! ensuring more reliable policy evaluation and testing.

/////////////////////////////////////////// MaskGenerator //////////////////////////////////////////

/// A pseudorandom mask generator for creating replacement values unlikely to be in LLM training data.
///
/// The MaskGenerator uses a linear congruential generator to produce deterministic
/// sequences of mask strings. Each mask is a fixed-length string using a reduced
/// character set designed to create values that are unlikely to appear in an LLM's
/// training data, avoiding visual confusion while maintaining uniqueness.
///
/// # Examples
///
/// ```
/// use policyai::MaskGenerator;
///
/// let mut generator = MaskGenerator::new();
/// let mask1 = generator.generate();
/// let mask2 = generator.generate();
/// assert_ne!(mask1, mask2);
/// ```
#[derive(Clone, Debug)]
pub struct MaskGenerator {
    /// Current position in the pseudorandom sequence.
    position: u128,
}

impl MaskGenerator {
    const LENGTH: usize = 6;
    const MODULUS: u128 = 3404825447;
    const PRIME: u128 = 3404825407;
    const BASE: usize = 23;
    const BASE23: [char; 23] = [
        'C', 'F', 'G', 'H', 'J', 'M', 'P', 'Q', 'R', 'V', 'W', 'X', 'c', 'f', 'g', 'h', 'j', 'm',
        'p', 'q', 'r', 'v', 'w',
    ];

    /// Create a new mask generator with a fixed seed.
    ///
    /// The generator starts at a predetermined position to ensure reproducible
    /// mask sequences across different runs.
    ///
    /// # Examples
    ///
    /// ```
    /// use policyai::MaskGenerator;
    ///
    /// let generator = MaskGenerator::new();
    /// ```
    pub const fn new() -> Self {
        Self {
            position: 743580272,
        }
    }

    /// Generate the next mask string in the sequence.
    ///
    /// Each generated mask is a fixed-length string using a carefully chosen
    /// character set designed to create values unlikely to be in LLM training data.
    /// The character set avoids visually similar characters, and the first character
    /// is always lowercase to ensure proper identifier formation.
    ///
    /// # Examples
    ///
    /// ```
    /// use policyai::MaskGenerator;
    ///
    /// let mut generator = MaskGenerator::new();
    /// let mask = generator.generate();
    /// assert_eq!(mask.len(), 6);
    /// assert!(mask.chars().next().unwrap().is_lowercase());
    /// ```
    pub fn generate(&mut self) -> String {
        let mut index = self.position;
        let mut s = String::with_capacity(Self::LENGTH);
        for i in 0..Self::LENGTH {
            let mut c = Self::BASE23[(index % Self::BASE as u128) as usize];
            if i == 0 {
                c = c.to_ascii_lowercase();
            }
            s.push(c);
            index /= Self::BASE as u128;
        }
        self.position = (self.position * Self::PRIME) % Self::MODULUS;
        s
    }
}

impl Default for MaskGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/////////////////////////////////////////////// tests //////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn first_ten_masks() {
        let mut masks = MaskGenerator::default();
        assert_eq!("fpXHcC", masks.generate());
        assert_eq!("pgXrqF", masks.generate());
        assert_eq!("fJpQVm", masks.generate());
        assert_eq!("vFRWmj", masks.generate());
        assert_eq!("rfwwgq", masks.generate());
        assert_eq!("gpjCvp", masks.generate());
        assert_eq!("ccpjVG", masks.generate());
        assert_eq!("hMmmFp", masks.generate());
        assert_eq!("pFFHvc", masks.generate());
        assert_eq!("jrGjMc", masks.generate());
    }

    #[test]
    fn at_least_one_thousand_policy_actions() {
        let mut masks = MaskGenerator::default();
        let mut seen: HashSet<String> = HashSet::default();
        for _ in 0..1024 {
            let mask = masks.generate();
            println!("checking {mask}");
            assert!(!seen.contains(&mask));
            let c = mask.chars().next().unwrap();
            assert!(c.is_lowercase());
            seen.insert(mask);
        }
    }
}
