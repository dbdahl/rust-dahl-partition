extern crate num_traits;

use crate::structure::*;
use crate::summary::psm::PairwiseSimilarityMatrixView;

use std::slice;

pub fn binder(
    partitions: &PartitionsHolderView,
    psm: &PairwiseSimilarityMatrixView,
    results: &mut [f64],
) {
    let ni = psm.n_items();
    for k in 0..partitions.n_partitions() {
        let mut sum = 0.0;
        for j in 0..ni {
            for i in 0..j {
                let p = unsafe { *psm.get_unchecked((i, j)) };
                sum += if unsafe {
                    *partitions.get_unchecked((k, i)) == *partitions.get_unchecked((k, j))
                } {
                    1.0 - p
                } else {
                    p
                }
            }
        }
        unsafe { *results.get_unchecked_mut(k) = sum };
    }
}

pub fn vilb(
    partitions: &PartitionsHolderView,
    psm: &PairwiseSimilarityMatrixView,
    results: &mut [f64],
) {
    let ni = psm.n_items();
    let sum2 = {
        let mut s1 = 0.0;
        for i in 0..ni {
            let mut s2 = 0.0;
            for j in 0..ni {
                s2 += unsafe { *psm.get_unchecked((i, j)) };
            }
            s1 += s2.log2()
        }
        s1
    };
    for k in 0..partitions.n_partitions() {
        let mut sum = sum2;
        for i in 0..ni {
            let mut s1 = 0u32;
            let mut s2 = 0.0;
            for j in 0..ni {
                if unsafe { *partitions.get_unchecked((k, i)) == *partitions.get_unchecked((k, j)) }
                {
                    s1 += 1;
                    s2 += unsafe { *psm.get_unchecked((i, j)) };
                }
            }
            sum += f64::from(s1).log2() - 2.0 * s2.log2();
        }
        unsafe { *results.get_unchecked_mut(k) = sum / (psm.n_items() as f64) };
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_partition__summary__expected_loss(
    n_partitions: i32,
    n_items: i32,
    partition_ptr: *mut i32,
    psm_ptr: *mut f64,
    loss: i32,
    results_ptr: *mut f64,
) {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let partitions = PartitionsHolderView::from_ptr(partition_ptr, np, ni, true);
    let psm = PairwiseSimilarityMatrixView::from_ptr(psm_ptr, ni);
    let results = slice::from_raw_parts_mut(results_ptr, np);
    match loss {
        0 => binder(&partitions, &psm, results),
        1 => vilb(&partitions, &psm, results),
        _ => panic!("Unsupported loss method: {}", loss),
    };
}
