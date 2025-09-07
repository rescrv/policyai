/////////////////////////////////////////// MaskGenerator //////////////////////////////////////////

#[derive(Clone, Debug)]
pub struct MaskGenerator {
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

    pub const fn new() -> Self {
        Self {
            position: 743580272,
        }
    }

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
