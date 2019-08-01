use crate::*;
use std::os::raw::{c_double, c_int};
use std::slice;

pub fn binder<A>(partition: &[A], psm: &PairwiseSimilarityMatrix) -> f64
where
    A: PartialEq,
{
    let ni = psm.n_items();
    let mut sum = 0.0;
    for j in 0..ni {
        for i in 0..j {
            let p = unsafe { *psm.get_unchecked((i, j)) };
            sum += if unsafe { *partition.get_unchecked(i) == *partition.get_unchecked(j) } {
                1.0 - p
            } else {
                p
            }
        }
    }
    sum
}

#[no_mangle]
pub unsafe extern "C" fn dahl_partition__summary__binder(
    n_items: c_int,
    partition_ptr: *const c_int,
    psm_ptr: *mut c_double,
) -> f64 {
    let ni = n_items as usize;
    let partition: &[c_int] = slice::from_raw_parts(partition_ptr, ni);
    let psm = PairwiseSimilarityMatrix::from_ptr(psm_ptr, ni);
    binder(partition, &psm)
}
