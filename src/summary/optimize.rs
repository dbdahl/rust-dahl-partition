extern crate num_cpus;
extern crate rand;

use crate::structure::*;
use crate::summary::loss::{
    binder_single, binder_single_partial, vilb_single, vilb_single_kernel, vilb_single_partial,
};
use crate::summary::psm::PairwiseSimilarityMatrixView;

use rand::seq::SliceRandom;
use rand::thread_rng;
use std::cmp::Ordering;
use std::convert::TryFrom;
use std::slice;
use std::sync::mpsc;

struct CacheUnit {
    item: usize,
    committed_sum: f64,
    committed_contribution: f64,
    speculative_sum: f64,
    speculative_contribution: f64,
}

pub struct VarOfInfoLBComputer<'a> {
    subsets: Vec<Vec<CacheUnit>>,
    psm: &'a PairwiseSimilarityMatrixView<'a>,
}

impl<'a> VarOfInfoLBComputer<'a> {
    pub fn new(psm: &'a PairwiseSimilarityMatrixView<'a>) -> VarOfInfoLBComputer<'a> {
        VarOfInfoLBComputer {
            subsets: Vec::new(),
            psm,
        }
    }

    pub fn new_subset(&mut self, partition: &mut Partition) {
        partition.new_subset();
        self.subsets.push(Vec::new())
    }

    pub fn look_ahead(&mut self, partition: &Partition, i: usize, subset_index: usize) -> f64 {
        let subset_of_partition = &partition.subsets()[subset_index];
        for cu in self.subsets[subset_index].iter_mut() {
            cu.speculative_sum = cu.committed_sum + self.psm[(cu.item, i)];
            cu.speculative_contribution = cu.speculative_sum.log2();
        }
        let sum = subset_of_partition
            .items()
            .iter()
            .fold(0.0, |s, j| s + self.psm[(i, *j)])
            + self.psm[(i, i)];
        self.subsets[subset_index].push(CacheUnit {
            item: i,
            committed_sum: 0.0,
            committed_contribution: 0.0,
            speculative_sum: sum,
            speculative_contribution: sum.log2(),
        });
        let nif = subset_of_partition.n_items() as f64;
        let s1 = if nif != 0.0 {
            (nif + 1.0) * (nif + 1.0).log2() - nif * nif.log2()
        } else {
            0.0
        };
        let s2 = self.subsets[subset_index].iter().fold(0.0, |s, cu| {
            s + cu.speculative_contribution - cu.committed_contribution
        });
        s1 - 2.0 * s2
    }

    pub fn add_with_index(&mut self, partition: &mut Partition, i: usize, subset_index: usize) {
        for (index, subset) in self.subsets.iter_mut().enumerate() {
            if index == subset_index {
                let cu = subset.last_mut().unwrap();
                cu.committed_sum = cu.speculative_sum;
                cu.committed_contribution = cu.speculative_contribution;
            } else {
                subset.pop();
            }
        }
        partition.add_with_index(i, subset_index);
    }
}

fn cmp_f64(a: &f64, b: &f64) -> Ordering {
    if a.is_nan() {
        return Ordering::Greater;
    }
    if b.is_nan() {
        return Ordering::Less;
    }
    if a < b {
        return Ordering::Less;
    } else if a > b {
        return Ordering::Greater;
    }
    return Ordering::Equal;
}

pub fn minimize_vilb_by_salso(
    max_size: usize,
    psm: &PairwiseSimilarityMatrixView,
    candidates: usize,
    _max_scans: usize,
    _parallel: bool,
) -> ((Vec<usize>, f64, usize), usize) {
    let ni = psm.n_items();
    let max_label = if max_size == 0 {
        usize::max_value()
    } else {
        max_size - 1
    };
    let mut global_minimum = std::f64::INFINITY;
    let mut global_best = Partition::new(ni);
    let mut global_n_scans = 0;
    let mut permutation: Vec<usize> = (0..ni).collect();
    let mut rng = thread_rng();
    for _ in 0..candidates {
        let mut vilb = VarOfInfoLBComputer::new(psm);
        let mut partition = Partition::new(ni);
        permutation.shuffle(&mut rng);
        for i in 0..ni {
            let ii = unsafe { *permutation.get_unchecked(i) };
            match partition.subsets().last() {
                None => vilb.new_subset(&mut partition),
                Some(last) => {
                    if !last.is_empty() && partition.n_subsets() <= max_label {
                        vilb.new_subset(&mut partition)
                    }
                }
            }
            let subset_index = (0..partition.n_subsets())
                .map(|subset_index| vilb.look_ahead(&mut partition, ii, subset_index))
                .enumerate()
                .min_by(|a, b| cmp_f64(&a.1, &b.1))
                .unwrap()
                .0;
            vilb.add_with_index(&mut partition, ii, subset_index);
        }
        let n_scans = 0;
        let value = vilb_single_kernel(&partition.labels()[..], psm);
        if value < global_minimum {
            global_minimum = value;
            global_best = partition;
            global_n_scans = n_scans;
        }
    }
    // Canonicalize the labels
    (
        (global_best.labels(), global_minimum, global_n_scans),
        candidates,
    )
}

pub fn minimize_by_salso(
    f: fn(&[usize], &[usize], usize, usize, &PairwiseSimilarityMatrixView) -> f64,
    g: fn(&[usize], &PairwiseSimilarityMatrixView) -> f64,
    max_size: usize,
    psm: &PairwiseSimilarityMatrixView,
    candidates: usize,
    max_scans: usize,
    parallel: bool,
) -> ((Vec<usize>, f64, usize), usize) {
    let max_label = if max_size == 0 {
        usize::max_value()
    } else {
        max_size - 1
    };
    if !parallel {
        (
            salso_engine(f, g, psm, candidates, max_label, max_scans),
            candidates,
        )
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
        (working_best, candidates * n_cores)
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
    max_scans: i32,
    loss: i32,
    max_size: i32,
    parallel: i32,
    results_labels_ptr: *mut i32,
    results_expected_loss_ptr: *mut f64,
    results_scans_ptr: *mut i32,
    results_actual_n_candidates_ptr: *mut i32,
) {
    let ni = usize::try_from(n_items).unwrap();
    let psm = PairwiseSimilarityMatrixView::from_ptr(psm_ptr, ni);
    let max_size = usize::try_from(max_size).unwrap();
    let candidates = usize::try_from(candidates).unwrap();
    let max_scans = usize::try_from(max_scans).unwrap();
    let parallel = parallel != 0;
    let ((minimizer, expected_loss, scans), actual_n_candidates) = match loss {
        0 => minimize_by_salso(
            binder_single_partial,
            binder_single,
            max_size,
            &psm,
            candidates,
            max_scans,
            parallel,
        ),
        1 => minimize_by_salso(
            vilb_single_partial,
            vilb_single,
            max_size,
            &psm,
            candidates,
            max_scans,
            parallel,
        ),
        2 => minimize_vilb_by_salso(max_size, &psm, candidates, max_scans, parallel),
        _ => panic!("Unsupported loss method: {}", loss),
    };
    let results_slice = slice::from_raw_parts_mut(results_labels_ptr, ni);
    for (i, v) in minimizer.iter().enumerate() {
        results_slice[i] = i32::try_from(*v).unwrap();
    }
    *results_expected_loss_ptr = expected_loss;
    *results_scans_ptr = i32::try_from(scans).unwrap();
    *results_actual_n_candidates_ptr = i32::try_from(actual_n_candidates).unwrap();
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
