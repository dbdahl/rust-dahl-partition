extern crate num_cpus;
extern crate rand;

use crate::structure::*;
use crate::summary::loss::{binder_single, binder_single_partitial, vilb_single_kernel};
use crate::summary::psm::PairwiseSimilarityMatrixView;

use rand::seq::SliceRandom;
use rand::thread_rng;
use std::convert::TryFrom;
use std::slice;
use std::sync::mpsc;

pub fn minimize_by_salso(
    f: fn(&[usize], &[usize], usize, &PairwiseSimilarityMatrixView) -> f64,
    g: fn(&[usize], &PairwiseSimilarityMatrixView) -> f64,
    psm: &PairwiseSimilarityMatrixView,
    n_candidates: usize,
) -> Vec<usize> {
    let ni = psm.n_items();
    let mut global_minimum = std::f64::INFINITY;
    let mut global_best: Vec<usize> = vec![0; ni];
    let mut partition: Vec<usize> = vec![0; ni];
    for _ in 0..n_candidates {
        let permutation = {
            let mut perm: Vec<usize> = (0..ni).collect();
            let mut rng = thread_rng();
            perm.shuffle(&mut rng);
            perm
        };
        let mut max: usize = 0;
        for n_allocated in 2..=ni {
            let ii = unsafe { *permutation.get_unchecked(n_allocated - 1) };
            let mut minimum = std::f64::INFINITY;
            let mut index = 0;
            for l in 0..=(max + 1) {
                partition[ii] = l;
                let value = f(&partition[..], &permutation[..], n_allocated, psm);
                if value < minimum {
                    minimum = value;
                    index = l;
                }
            }
            if index > max {
                max = index;
            }
            partition[ii] = index;
        }
        let value = g(&partition[..], psm);
        if value < global_minimum {
            global_minimum = value;
            global_best = partition.clone();
        }
    }
    global_best
}

pub fn minimize_by_enumeration(
    f: fn(&[usize], &PairwiseSimilarityMatrixView) -> f64,
    psm: &PairwiseSimilarityMatrixView,
) -> Vec<usize> {
    let (tx, rx) = mpsc::channel();
    crossbeam::scope(|s| {
        for iter in Partition::iter_sharded(num_cpus::get() as u32, psm.n_items()) {
            let tx = mpsc::Sender::clone(&tx);
            s.spawn(move |_| {
                let mut working_minimum = std::f64::INFINITY;
                let mut working_minimizer = vec![0usize; psm.n_items()];
                for partition in iter {
                    let value = f(&partition[..], psm);
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
        let value = f(&partition[..], psm);
        if value < working_minimum {
            working_minimum = value;
            working_minimizer = partition;
        }
    }
    working_minimizer
}

#[no_mangle]
pub unsafe extern "C" fn dahl_partition__summary__minimize_by_salso(
    n_items: i32,
    psm_ptr: *mut f64,
    n_candidates: usize,
    loss: i32,
    results_ptr: *mut i32,
) {
    let ni = usize::try_from(n_items).unwrap();
    let psm = PairwiseSimilarityMatrixView::from_ptr(psm_ptr, ni);
    let (f, g) = match loss {
        0 => (binder_single_partitial, binder_single),
        //1 => vilb_single_partital,
        _ => panic!("Unsupported loss method: {}", loss),
    };
    let minimizer = minimize_by_salso(f, g, &psm, n_candidates);
    let results_slice = slice::from_raw_parts_mut(results_ptr, ni);
    for (i, v) in minimizer.iter().enumerate() {
        results_slice[i] = i32::try_from(*v).unwrap();
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_partition__summary__minimize_by_enumeration(
    n_items: i32,
    psm_ptr: *mut f64,
    loss: i32,
    results_ptr: *mut i32,
) {
    let ni = usize::try_from(n_items).unwrap();
    let psm = PairwiseSimilarityMatrixView::from_ptr(psm_ptr, ni);
    let f = match loss {
        0 => binder_single,
        1 => vilb_single_kernel,
        _ => panic!("Unsupported loss method: {}", loss),
    };
    let minimizer = minimize_by_enumeration(f, &psm);
    let results_slice = slice::from_raw_parts_mut(results_ptr, ni);
    for (i, v) in minimizer.iter().enumerate() {
        results_slice[i] = i32::try_from(*v).unwrap();
    }
}
