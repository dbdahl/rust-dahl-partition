extern crate num_bigint;
extern crate num_traits;

use num_bigint::BigUint;
use num_traits::cast::ToPrimitive;
use num_traits::{One, Zero};
use std::convert::TryFrom;
use std::os::raw::c_int;

pub fn lbell(n: usize) -> f64 {
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
    let value = &r1[n - 1];
    let blex = i64::try_from(value.bits()).unwrap() - 1022;
    if blex > 0 {
        let y: f64 = (value >> (blex as usize)).to_f64().unwrap();
        let w: f64 = 2.0;
        y + (blex as f64) * w.ln()
    } else {
        value.to_f64().unwrap().ln()
    }
}

#[no_mangle]
pub extern "C" fn dahl_partition__utils__lbell(n: c_int) -> f64 {
    let n = usize::try_from(n).unwrap();
    lbell(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lbell() {
        assert!((lbell(5).exp() - 52.0).abs() < 1.0)
    }

}
