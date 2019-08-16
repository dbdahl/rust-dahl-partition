extern crate num_bigint;

use num_bigint::BigUint;
use num_traits::cast::ToPrimitive;
use num_traits::{One, Zero};
use std::convert::TryFrom;
use std::f64;

/// Computes the natural logarithm of the Bell number.
///
pub fn lbell(n: usize) -> f64 {
    let value = bell(n);
    let n_bits = value.bits();
    let threshold = 1022usize;
    let log2 = if n_bits > threshold {
        let n_shifted_bits = value.bits() - threshold;
        let shifted_value = value >> n_shifted_bits;
        if shifted_value.bits() > threshold {
            return f64::INFINITY;
        }
        let y: f64 = shifted_value.to_f64().unwrap();
        (n_shifted_bits as f64) + y.log2()
    } else {
        value.to_f64().unwrap().log2()
    };
    log2 / f64::consts::LOG2_E
}

pub fn bell(n: usize) -> BigUint {
    let mut r1: Vec<BigUint> = vec![Zero::zero(); n];
    let mut r2: Vec<BigUint> = vec![Zero::zero(); n];
    r1[0] = One::one();
    for k in 1..n {
        r2[0] = r1[k - 1].clone();
        for i in 1..(k + 1) {
            r2[i] = r1[i - 1].clone() + &r2[i - 1];
        }
        let tmp = r1;
        r1 = r2;
        r2 = tmp;
    }
    r1[n - 1].clone()
}

#[no_mangle]
pub extern "C" fn dahl_partition__utils__lbell(n: i32) -> f64 {
    if n < 0 {
        return 0.0;
    }
    match usize::try_from(n) {
        Ok(n) => lbell(n),
        Err(_) => f64::INFINITY,
    }
}

#[no_mangle]
pub extern "C" fn dahl_partition__utils__bell(n: i32) -> f64 {
    if n < 0 {
        return 0.0;
    }
    match usize::try_from(n) {
        Ok(n) => match bell(n).to_f64() {
            Some(v) => v,
            None => f64::INFINITY,
        },
        Err(_) => f64::INFINITY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lbell() {
        relative_eq!(lbell(220), 714.4033);
        relative_eq!(bell(5).to_f64().unwrap(), 52.0);
    }

}
