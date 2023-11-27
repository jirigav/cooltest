use crate::common::{bit_value_in_block, z_score};
use std::{collections::HashSet, fmt::Debug};

pub(crate) trait Distinguisher {
    fn evaluate(&self, block: &[u8]) -> bool;

    fn forget_count(&mut self);

    fn increase_count(&mut self, n: usize);

    fn get_count(&self) -> usize;

    fn z_score(&mut self, samples: usize) -> f64;

    fn get_z_score(&self) -> Option<f64>;
}

pub(crate) fn evaluate_distinguisher<P: Distinguisher + ?Sized>(
    distinguisher: &mut P,
    data: &[Vec<u8>],
) -> f64 {
    distinguisher.forget_count();
    distinguisher.increase_count(
        data.iter()
            .filter(|block| distinguisher.evaluate(block))
            .count(),
    );
    distinguisher.z_score(data.len())
}

#[derive(Debug)]
pub(crate) struct Polynomial {
    pub(crate) monomials: Vec<Vec<usize>>,
    used_variables: HashSet<usize>,
    pub(crate) probability: f64,
    pub(crate) z_score: Option<f64>,
    count: Option<usize>,
}

impl Polynomial {
    pub(crate) fn new() -> Polynomial {
        Polynomial {
            monomials: vec![Vec::new()],
            used_variables: HashSet::new(),
            probability: 1.0,
            z_score: None,
            count: None,
        }
    }

    fn sort(&mut self) {
        self.monomials.sort_unstable();
        self.monomials.iter_mut().for_each(|x| x.sort_unstable());
    }

    pub(crate) fn negate(&mut self) {
        self.probability = 1.0 - self.probability;
        self.xor(Vec::new());
        self.sort();
    }

    pub(crate) fn contains(&self, variable: usize) -> bool {
        self.used_variables.contains(&variable)
    }

    fn xor(&mut self, monomial: Vec<usize>) {
        if self.monomials.contains(&monomial) {
            self.monomials.retain(|x| *x != monomial);
        } else {
            self.monomials.push(monomial);
        }
        self.sort();
    }

    pub(crate) fn and(&mut self, variable: usize, negated: bool) -> Result<(), ()> {
        if self.used_variables.contains(&variable) {
            return Err(());
        }
        self.used_variables.insert(variable);
        self.probability *= 0.5;
        if negated {
            let mut ab = self.monomials.clone();
            ab.iter_mut().for_each(|x| x.push(variable));
            self.monomials.append(&mut ab);
        } else {
            self.monomials.iter_mut().for_each(|x| x.push(variable));
        }
        self.sort();
        Ok(())
    }
}
impl Distinguisher for Polynomial {
    fn evaluate(&self, block: &[u8]) -> bool {
        let mut result = false;
        for m in &self.monomials {
            result ^= m.iter().all(|x| bit_value_in_block(*x, block));
        }
        result
    }

    fn forget_count(&mut self) {
        self.count = None;
    }

    fn increase_count(&mut self, n: usize) {
        if let Some(count) = self.count {
            self.count = Some(count + n);
        } else {
            self.count = Some(n);
        }
    }

    fn get_count(&self) -> usize {
        if let Some(count) = self.count {
            count
        } else {
            0
        }
    }

    fn z_score(&mut self, sample_size: usize) -> f64 {
        self.z_score = Some(z_score(sample_size, self.get_count(), self.probability));
        self.z_score.unwrap()
    }

    fn get_z_score(&self) -> Option<f64> {
        self.z_score
    }
}

impl PartialEq for Polynomial {
    fn eq(&self, other: &Polynomial) -> bool {
        self.monomials == other.monomials
    }
}

impl Clone for Polynomial {
    fn clone(&self) -> Self {
        Polynomial {
            monomials: self.monomials.clone(),
            used_variables: self.used_variables.clone(),
            probability: self.probability,
            z_score: None,
            count: None,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::Polynomial;

    #[test]
    fn polynomial_test() {
        let mut poly = Polynomial::new();
        poly.and(2, true).unwrap();
        assert_eq!(poly.monomials, vec![Vec::new(), vec![2]]);
        poly.and(1, false).unwrap();
        assert_eq!(poly.monomials, vec![vec![1], vec![2, 1]]);
        poly.negate();
        assert_eq!(poly.monomials, vec![vec![1], vec![2, 1], Vec::new()]);
        assert_eq!(poly.probability, 0.75);
        assert!(poly.and(2, false).is_err())
    }
}
