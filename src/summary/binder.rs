use crate::*;
use std::os::raw::{c_double, c_int};
use std::slice;

pub fn binder_paired(p: f64) -> f64 {
    1.0 - p
}

pub fn binder_unpaired(p: f64) -> f64 {
    p
}

pub fn expected_loss<A>(
    partitions: &PartitionsHolderView<A>,
    psm: &PairwiseSimilarityMatrixView,
    paired: fn(f64) -> f64,
    unpaired: fn(f64) -> f64,
    results: &mut [f64],
) where
    A: PartialEq,
{
    let ni = psm.n_items();
    for k in 0..partitions.n_samples {
        let mut sum = 0.0;
        for j in 0..ni {
            for i in 0..j {
                let p = unsafe { *psm.get_unchecked((i, j)) };
                sum += if unsafe {
                    *partitions.get_unchecked((k, i)) == *partitions.get_unchecked((k, j))
                } {
                    paired(p)
                } else {
                    unpaired(p)
                }
            }
        }
        unsafe { *results.get_unchecked_mut(k) = sum };
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_partition__summary__expected_loss(
    n_samples: c_int,
    n_items: c_int,
    partition_ptr: *const c_int,
    psm_ptr: *mut c_double,
    loss: c_int,
    results_ptr: *mut c_double,
) {
    let ns = n_samples as usize;
    let ni = n_items as usize;
    let partitions = PartitionsHolderView::from_ptr(partition_ptr, ns, ni, true);
    let psm = PairwiseSimilarityMatrixView::from_ptr(psm_ptr, ni);
    let results = slice::from_raw_parts_mut(results_ptr, ns);
    match loss {
        0 => expected_loss(&partitions, &psm, binder_paired, binder_unpaired, results),
        0 => expected_loss(&partitions, &psm, binder_paired, binder_unpaired, results),
        _ => panic!("Unsupported loss method: {}", loss),
    };
}
