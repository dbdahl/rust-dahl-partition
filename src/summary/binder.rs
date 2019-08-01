// use crate::*;
use std::os::raw::{c_double, c_int};
use std::slice;

pub fn binder<A>(partition: &[A], epam: &[f64]) -> f64
where
    A: PartialEq,
{
    let ni = partition.len();
    let mut sum = 0.0;
    for j in 0..ni {
        let nij = ni * j;
        for i in 0..j {
            let p = unsafe { *epam.get_unchecked(nij + i) };
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
    epam_ptr: *mut c_double,
) -> f64 {
    let ni = n_items as usize;
    let partition: &[c_int] = slice::from_raw_parts(partition_ptr, ni);
    let epam: &mut [c_double] = slice::from_raw_parts_mut(epam_ptr, ni * ni);
    binder(partition, epam)
}
