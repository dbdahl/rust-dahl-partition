use crate::*;
use std::os::raw::{c_double, c_int};
use std::slice;

pub fn binder<A>(
    partitions: &PartitionsHolderView<A>,
    psm: &PairwiseSimilarityMatrixView,
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
                    1.0 - p
                } else {
                    p
                }
            }
        }
        unsafe { *results.get_unchecked_mut(k) = sum };
    }
}

pub fn vilb<A>(
    partitions: &PartitionsHolderView<A>,
    psm: &PairwiseSimilarityMatrixView,
    results: &mut [f64],
) where
    A: PartialEq,
{
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
    for k in 0..partitions.n_samples {
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
        unsafe { *results.get_unchecked_mut(k) = sum / (psm.n_items as f64) };
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
        0 => binder(&partitions, &psm, results),
        1 => vilb(&partitions, &psm, results),
        _ => panic!("Unsupported loss method: {}", loss),
    };
}
