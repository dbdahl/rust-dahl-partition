use crate::structure::*;
use crate::summary::loss::binder_single;
use crate::summary::psm::PairwiseSimilarityMatrixView;

use std::convert::TryFrom;
use std::slice;

pub fn minimize_binder_by_enumeration(psm: &PairwiseSimilarityMatrixView) -> (Vec<usize>, f64) {
    let mut working_minimum = std::f64::INFINITY;
    let mut working_minimizer = vec![0usize; psm.n_items()];
    for partition in Partition::iter(psm.n_items()) {
        let value = binder_single(&partition[..], psm);
        if value < working_minimum {
            working_minimum = value;
            working_minimizer = partition;
        }
    }
    (working_minimizer, working_minimum)
}

pub fn minimize_vilb_by_enumeration(psm: &PairwiseSimilarityMatrixView) -> (Vec<usize>, f64) {
    minimize_binder_by_enumeration(psm)
}

#[no_mangle]
pub unsafe extern "C" fn dahl_partition__summary__minimize_by_enumeration(
    n_items: i32,
    psm_ptr: *mut f64,
    loss: i32,
    results_ptr: *mut i32,
    results_value_ptr: *mut f64,
) {
    let ni = n_items as usize;
    let psm = PairwiseSimilarityMatrixView::from_ptr(psm_ptr, ni);
    let (minimizer, minimum) = match loss {
        0 => minimize_binder_by_enumeration(&psm),
        1 => minimize_vilb_by_enumeration(&psm),
        _ => panic!("Unsupported loss method: {}", loss),
    };
    let results_slice = slice::from_raw_parts_mut(results_ptr, ni);
    for (i, v) in minimizer.iter().enumerate() {
        results_slice[i] = i32::try_from(*v).unwrap();
    }
    *results_value_ptr = minimum;
}
