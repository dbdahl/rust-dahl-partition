extern crate num_cpus;

use crate::structure::*;
use crate::summary::loss::{binder_single, vilb_single_kernel};
use crate::summary::psm::PairwiseSimilarityMatrixView;

use std::convert::TryFrom;
use std::slice;
use std::sync::mpsc;

pub fn minimize_binder_by_enumeration(psm: &PairwiseSimilarityMatrixView) -> Vec<usize> {
    let mut working_minimum = std::f64::INFINITY;
    let mut working_minimizer = vec![0usize; psm.n_items()];
    for partition in Partition::iter(psm.n_items()) {
        let value = binder_single(&partition[..], psm);
        if value < working_minimum {
            working_minimum = value;
            working_minimizer = partition;
        }
    }
    working_minimizer
}

pub fn minimize_vilb_by_enumeration(psm: &PairwiseSimilarityMatrixView<'_>) -> Vec<usize> {
    let (tx, rx) = mpsc::channel();
    crossbeam::scope(|s| {
        for iter in Partition::iter_sharded(num_cpus::get() as u32, psm.n_items()) {
            let tx = mpsc::Sender::clone(&tx);
            s.spawn(move |_| {
                let mut working_minimum = std::f64::INFINITY;
                let mut working_minimizer = vec![0usize; psm.n_items()];
                for partition in iter {
                    let value = vilb_single_kernel(&partition[..], psm);
                    if value < working_minimum {
                        working_minimum = value;
                        working_minimizer = partition;
                    }
                }
                tx.send(working_minimizer).unwrap();
            });
        }
    })
    .unwrap();
    std::mem::drop(tx); // Because of the cloning in the loop.
    let mut working_minimum = std::f64::INFINITY;
    let mut working_minimizer = vec![0usize; psm.n_items()];
    for partition in rx {
        let value = vilb_single_kernel(&partition[..], psm);
        if value < working_minimum {
            working_minimum = value;
            working_minimizer = partition;
        }
    }
    working_minimizer
}

#[no_mangle]
pub unsafe extern "C" fn dahl_partition__summary__minimize_by_enumeration(
    n_items: i32,
    psm_ptr: *mut f64,
    loss: i32,
    results_ptr: *mut i32,
) {
    let ni = n_items as usize;
    let psm = PairwiseSimilarityMatrixView::from_ptr(psm_ptr, ni);
    let minimizer = match loss {
        0 => minimize_binder_by_enumeration(&psm),
        1 => minimize_vilb_by_enumeration(&psm),
        _ => panic!("Unsupported loss method: {}", loss),
    };
    let results_slice = slice::from_raw_parts_mut(results_ptr, ni);
    for (i, v) in minimizer.iter().enumerate() {
        results_slice[i] = i32::try_from(*v).unwrap();
    }
}
