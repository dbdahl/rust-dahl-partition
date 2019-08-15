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
    f: fn(&[usize], &[usize], usize, usize, &PairwiseSimilarityMatrixView) -> f64,
    g: fn(&[usize], &PairwiseSimilarityMatrixView) -> f64,
    psm: &PairwiseSimilarityMatrixView,
    candidates: usize,
    max_size: usize,
    max_scans: usize,
    parallel: bool,
) -> (Vec<usize>, f64, usize) {
    let max_label = if max_size == 0 {
        usize::max_value()
    } else {
        max_size - 1
    };
    if !parallel {
        salso_engine(f, g, psm, candidates, max_label, max_scans)
    } else {
        let (tx, rx) = mpsc::channel();
        let n_cores = num_cpus::get();
        let candidates = (candidates + n_cores - 1) / n_cores;
        crossbeam::scope(|s| {
            for _ in 0..n_cores {
                let tx = mpsc::Sender::clone(&tx);
                s.spawn(move |_| {
                    tx.send(salso_engine(f, g, psm, candidates, max_label, max_scans))
                        .unwrap();
                });
            }
        })
        .unwrap();
        std::mem::drop(tx); // Because of the cloning in the loop.
        let mut working_best = (vec![0usize; psm.n_items()], std::f64::INFINITY, 0);
        for candidate in rx {
            if candidate.1 < working_best.1 {
                working_best = candidate;
            }
        }
        working_best
    }
}

pub fn salso_engine(
    f: fn(&[usize], &[usize], usize, usize, &PairwiseSimilarityMatrixView) -> f64,
    g: fn(&[usize], &PairwiseSimilarityMatrixView) -> f64,
    psm: &PairwiseSimilarityMatrixView,
    candidates: usize,
    max_label: usize,
    max_scans: usize,
) -> (Vec<usize>, f64, usize) {
    let ni = psm.n_items();
    let mut global_minimum = std::f64::INFINITY;
    let mut global_best: Vec<usize> = vec![0; ni];
    let mut global_n_scans = 0;
    let mut partition: Vec<usize> = vec![0; ni];
    let mut permutation: Vec<usize> = (0..ni).collect();
    let mut rng = thread_rng();
    for _ in 0..candidates {
        permutation.shuffle(&mut rng);
        // Initial allocation
        partition[unsafe { *permutation.get_unchecked(0) }] = 0;
        let mut max: usize = 0;
        for n_allocated in 2..=ni {
            let ii = unsafe { *permutation.get_unchecked(n_allocated - 1) };
            let mut minimum = std::f64::INFINITY;
            let mut index = 0;
            for l in 0..=(max + 1).min(max_label) {
                partition[ii] = l;
                let value = f(
                    &partition[..],
                    &permutation[..],
                    n_allocated - 1,
                    n_allocated,
                    psm,
                );
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
        // Sweetening scans
        let mut n_scans = max_scans;
        for scan in 0..max_scans {
            let previous = partition.clone();
            for i in 0..ni {
                let ii = unsafe { *permutation.get_unchecked(i) };
                let mut minimum = std::f64::INFINITY;
                let mut index = 0;
                for l in 0..=(max + 1).min(max_label) {
                    partition[ii] = l;
                    let value = f(&partition[..], &permutation[..], i, ni, psm);
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
            if partition == previous {
                n_scans = scan + 1;
                break;
            }
        }
        let value = g(&partition[..], psm);
        if value < global_minimum {
            global_minimum = value;
            global_best = partition.clone();
            global_n_scans = n_scans;
        }
    }
    // Canonicalize the labels
    (
        Partition::from(&global_best[..]).labels(),
        global_minimum,
        global_n_scans,
    )
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
    candidates: i32,
    max_size: i32,
    max_scans: i32,
    loss: i32,
    parallel: i32,
    results_labels_ptr: *mut i32,
    results_expected_loss_ptr: *mut f64,
    results_scans_ptr: *mut i32,
) {
    let ni = usize::try_from(n_items).unwrap();
    let psm = PairwiseSimilarityMatrixView::from_ptr(psm_ptr, ni);
    let (f, g) = match loss {
        0 => (binder_single_partitial, binder_single),
        //1 => vilb_single_partital,
        _ => panic!("Unsupported loss method: {}", loss),
    };
    let (minimizer, expected_loss, scans) = minimize_by_salso(
        f,
        g,
        &psm,
        usize::try_from(candidates).unwrap(),
        usize::try_from(max_size).unwrap(),
        usize::try_from(max_scans).unwrap(),
        parallel != 0,
    );
    let results_slice = slice::from_raw_parts_mut(results_labels_ptr, ni);
    for (i, v) in minimizer.iter().enumerate() {
        results_slice[i] = i32::try_from(*v).unwrap();
    }
    *results_expected_loss_ptr = expected_loss;
    *results_scans_ptr = i32::try_from(scans).unwrap();
}

#[no_mangle]
pub unsafe extern "C" fn dahl_partition__summary__minimize_by_enumeration(
    n_items: i32,
    psm_ptr: *mut f64,
    loss: i32,
    results_label_ptr: *mut i32,
) {
    let ni = usize::try_from(n_items).unwrap();
    let psm = PairwiseSimilarityMatrixView::from_ptr(psm_ptr, ni);
    let f = match loss {
        0 => binder_single,
        1 => vilb_single_kernel,
        _ => panic!("Unsupported loss method: {}", loss),
    };
    let minimizer = minimize_by_enumeration(f, &psm);
    let results_slice = slice::from_raw_parts_mut(results_label_ptr, ni);
    for (i, v) in minimizer.iter().enumerate() {
        results_slice[i] = i32::try_from(*v).unwrap();
    }
}
