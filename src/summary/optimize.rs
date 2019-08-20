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

fn cmp_f64(a: &f64, b: &f64) -> Ordering {
    if a.is_nan() {
        Ordering::Greater
    } else if b.is_nan() {
        Ordering::Less
    } else if a < b {
        Ordering::Less
    } else if a > b {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

// Binder Loss

#[derive(Debug)]
struct BinderSubsetCalculations {
    committed_loss: f64,
    speculative_loss: f64,
}

pub struct BinderComputer<'a> {
    subsets: Vec<BinderSubsetCalculations>,
    psm: &'a PairwiseSimilarityMatrixView<'a>,
}

impl<'a> BinderComputer<'a> {
    pub fn new(psm: &'a PairwiseSimilarityMatrixView<'a>) -> BinderComputer<'a> {
        BinderComputer {
            subsets: Vec::new(),
            psm,
        }
    }

    pub fn new_subset(&mut self, partition: &mut Partition) {
        partition.new_subset();
        self.subsets.push(BinderSubsetCalculations {
            committed_loss: 0.0,
            speculative_loss: 0.0,
        })
    }

    pub fn speculative_add(&mut self, partition: &Partition, i: usize, subset_index: usize) -> f64 {
        self.subsets[subset_index].speculative_loss =
            partition
                .subsets()
                .iter()
                .enumerate()
                .fold(0.0, |s, (index, subset)| {
                    s + if index == subset_index {
                        subset.items().iter().fold(0.0, |s, j| {
                            s + 1.0 - unsafe { *self.psm.get_unchecked((i, *j)) }
                        })
                    } else {
                        subset
                            .items()
                            .iter()
                            .fold(0.0, |s, j| s + unsafe { *self.psm.get_unchecked((i, *j)) })
                    }
                });
        self.subsets[subset_index].speculative_loss
    }

    pub fn add_with_index(&mut self, partition: &mut Partition, i: usize, subset_index: usize) {
        self.subsets[subset_index].committed_loss += self.subsets[subset_index].speculative_loss;
        partition.add_with_index(i, subset_index);
    }

    pub fn remove(&mut self, partition: &mut Partition, i: usize) -> usize {
        let subset_index = partition.label_of(i).unwrap();
        partition.remove_with_index(i, subset_index);
        partition.clean_subset(subset_index);
        self.subsets[subset_index].committed_loss =
            -partition
                .subsets()
                .iter()
                .enumerate()
                .fold(0.0, |s, (index, subset)| {
                    s + if index == subset_index {
                        subset.items().iter().fold(0.0, |s, j| {
                            s + (1.0 - unsafe { *self.psm.get_unchecked((i, *j)) })
                        })
                    } else {
                        subset
                            .items()
                            .iter()
                            .fold(0.0, |s, j| s + unsafe { *self.psm.get_unchecked((i, *j)) })
                    }
                });

        subset_index
    }

    pub fn expected_loss(&mut self) -> f64 {
        self.expected_loss_unnormalized()
    }

    pub fn expected_loss_unnormalized(&mut self) -> f64 {
        self.subsets
            .iter()
            .fold(0.0, |s, subset| s + subset.committed_loss)
    }
}

fn binder_ensure_empty_subset(
    partition: &mut Partition,
    binder: &mut BinderComputer,
    max_label: usize,
) {
    match partition.subsets().last() {
        None => binder.new_subset(partition),
        Some(last) => {
            if !last.is_empty() && partition.n_subsets() <= max_label {
                binder.new_subset(partition)
            }
        }
    }
}

fn binder_micro_optimized_allocation(
    partition: &mut Partition,
    binder: &mut BinderComputer,
    i: usize,
) -> usize {
    let subset_index = (0..partition.n_subsets())
        .map(|subset_index| binder.speculative_add(partition, i, subset_index))
        .enumerate()
        .min_by(|a, b| cmp_f64(&a.1, &b.1))
        .unwrap()
        .0;
    binder.add_with_index(partition, i, subset_index);
    subset_index
}

pub fn minimize_binder_by_salso(
    max_size: usize,
    psm: &PairwiseSimilarityMatrixView,
    candidates: usize,
    max_scans: usize,
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
        let mut binder = BinderComputer::new(psm);
        let mut partition = Partition::new(ni);
        permutation.shuffle(&mut rng);
        // Initial allocation
        for i in 0..ni {
            binder_ensure_empty_subset(&mut partition, &mut binder, max_label);
            let ii = unsafe { *permutation.get_unchecked(i) };
            binder_micro_optimized_allocation(&mut partition, &mut binder, ii);
        }
        // Sweetening scans
        let mut n_scans = max_scans;
        for scan in 0..max_scans {
            let mut no_change = true;
            for i in 0..ni {
                binder_ensure_empty_subset(&mut partition, &mut binder, max_label);
                let ii = unsafe { *permutation.get_unchecked(i) };
                let previous_subset_index = binder.remove(&mut partition, ii);
                let subset_index =
                    binder_micro_optimized_allocation(&mut partition, &mut binder, ii);
                if subset_index != previous_subset_index {
                    no_change = false;
                };
            }
            if no_change {
                n_scans = scan + 1;
                break;
            }
        }
        let value = binder.expected_loss_unnormalized();
        if value < global_minimum {
            global_minimum = value;
            global_best = partition;
            global_n_scans = n_scans;
        }
    }
    // Canonicalize the labels
    global_best.canonicalize();
    let labels = global_best.labels_via_copying();
    let loss = global_minimum;
    ((labels, loss, global_n_scans), candidates)
}

// Variation of Information Loss

#[derive(Debug)]
struct VarOfInforLBCacheUnit {
    item: usize,
    committed_sum: f64,
    committed_contribution: f64,
    speculative_sum: f64,
    speculative_contribution: f64,
}

#[derive(Debug)]
struct VarOfInfoLBSubsetCalculations {
    cached_units: Vec<VarOfInforLBCacheUnit>,
    committed_loss: f64,
    speculative_loss: f64,
}

pub struct VarOfInfoLBComputer<'a> {
    subsets: Vec<VarOfInfoLBSubsetCalculations>,
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
        self.subsets.push(VarOfInfoLBSubsetCalculations {
            cached_units: Vec::new(),
            committed_loss: 0.0,
            speculative_loss: 0.0,
        })
    }

    pub fn speculative_add(&mut self, partition: &Partition, i: usize, subset_index: usize) -> f64 {
        let subset_of_partition = &partition.subsets()[subset_index];
        if subset_of_partition.n_items() == 0 {
            self.subsets[subset_index]
                .cached_units
                .push(VarOfInforLBCacheUnit {
                    item: i,
                    committed_sum: 0.0,
                    committed_contribution: 0.0,
                    speculative_sum: 1.0,
                    speculative_contribution: 0.0,
                });
            return 0.0;
        }
        for cu in self.subsets[subset_index].cached_units.iter_mut() {
            cu.speculative_sum =
                cu.committed_sum + unsafe { *self.psm.get_unchecked((cu.item, i)) };
            cu.speculative_contribution = cu.speculative_sum.log2();
        }
        let sum = subset_of_partition
            .items()
            .iter()
            .fold(0.0, |s, j| s + unsafe { *self.psm.get_unchecked((i, *j)) })
            + 1.0; // Because self.psm[(i, i)] == 1;
        self.subsets[subset_index]
            .cached_units
            .push(VarOfInforLBCacheUnit {
                item: i,
                committed_sum: 0.0,
                committed_contribution: 0.0,
                speculative_sum: sum,
                speculative_contribution: sum.log2(),
            });
        let nif = subset_of_partition.n_items() as f64;
        let s1 = (nif + 1.0) * (nif + 1.0).log2();
        let s2 = self.subsets[subset_index]
            .cached_units
            .iter()
            .fold(0.0, |s, cu| s + cu.speculative_contribution);
        self.subsets[subset_index].speculative_loss = s1 - 2.0 * s2;
        self.subsets[subset_index].speculative_loss - self.subsets[subset_index].committed_loss
    }

    pub fn add_with_index(&mut self, partition: &mut Partition, i: usize, subset_index: usize) {
        for (index, subset) in self.subsets.iter_mut().enumerate() {
            if index == subset_index {
                for cu in subset.cached_units.iter_mut() {
                    cu.committed_sum = cu.speculative_sum;
                    cu.committed_contribution = cu.speculative_contribution;
                }
            } else {
                subset.cached_units.pop();
            }
        }
        self.subsets[subset_index].committed_loss = self.subsets[subset_index].speculative_loss;
        partition.add_with_index(i, subset_index);
    }

    pub fn remove(&mut self, partition: &mut Partition, i: usize) -> usize {
        let subset_index = partition.label_of(i).unwrap();
        for cu in self.subsets[subset_index].cached_units.iter_mut() {
            cu.committed_sum -= unsafe { self.psm.get_unchecked((cu.item, i)) };
            cu.committed_contribution = cu.committed_sum.log2();
        }
        let pos = self.subsets[subset_index]
            .cached_units
            .iter()
            .enumerate()
            .find(|cu| cu.1.item == i)
            .unwrap()
            .0;
        self.subsets[subset_index].cached_units.swap_remove(pos);
        self.subsets[subset_index].committed_loss =
            match partition.subsets()[subset_index].n_items() {
                0 => 0.0,
                ni => {
                    let nif = ni as f64;
                    nif * nif.log2()
                        - 2.0
                            * self.subsets[subset_index]
                                .cached_units
                                .iter()
                                .fold(0.0, |s, cu| s + cu.committed_contribution)
                }
            };
        partition.remove_with_index(i, subset_index);
        partition.clean_subset(subset_index);
        subset_index
    }

    pub fn expected_loss(&mut self) -> f64 {
        let nif = self.psm.n_items() as f64;
        (self.expected_loss_unnormalized() + Self::expected_loss_constant(self.psm)) / nif
    }

    pub fn expected_loss_unnormalized(&mut self) -> f64 {
        self.subsets
            .iter()
            .fold(0.0, |s, subset| s + subset.committed_loss)
    }

    pub fn expected_loss_constant(psm: &PairwiseSimilarityMatrixView) -> f64 {
        let ni = psm.n_items();
        let mut s1: f64 = 0.0;
        for i in 0..ni {
            let mut s2: f64 = 0.0;
            for j in 0..ni {
                s2 += unsafe { psm.get_unchecked((i, j)) };
            }
            s1 += s2.log2();
        }
        s1
    }
}

fn vilb_ensure_empty_subset(
    partition: &mut Partition,
    vilb: &mut VarOfInfoLBComputer,
    max_label: usize,
) {
    match partition.subsets().last() {
        None => vilb.new_subset(partition),
        Some(last) => {
            if !last.is_empty() && partition.n_subsets() <= max_label {
                vilb.new_subset(partition)
            }
        }
    }
}

fn vilb_micro_optimized_allocation(
    partition: &mut Partition,
    vilb: &mut VarOfInfoLBComputer,
    i: usize,
) -> usize {
    let subset_index = (0..partition.n_subsets())
        .map(|subset_index| vilb.speculative_add(partition, i, subset_index))
        .enumerate()
        .min_by(|a, b| cmp_f64(&a.1, &b.1))
        .unwrap()
        .0;
    vilb.add_with_index(partition, i, subset_index);
    subset_index
}

pub fn minimize_vilb_by_salso(
    max_size: usize,
    psm: &PairwiseSimilarityMatrixView,
    candidates: usize,
    max_scans: usize,
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
        // Initial allocation
        for i in 0..ni {
            vilb_ensure_empty_subset(&mut partition, &mut vilb, max_label);
            let ii = unsafe { *permutation.get_unchecked(i) };
            vilb_micro_optimized_allocation(&mut partition, &mut vilb, ii);
        }
        // Sweetening scans
        let mut n_scans = max_scans;
        for scan in 0..max_scans {
            let mut no_change = true;
            for i in 0..ni {
                vilb_ensure_empty_subset(&mut partition, &mut vilb, max_label);
                let ii = unsafe { *permutation.get_unchecked(i) };
                let previous_subset_index = vilb.remove(&mut partition, ii);
                let subset_index = vilb_micro_optimized_allocation(&mut partition, &mut vilb, ii);
                if subset_index != previous_subset_index {
                    no_change = false;
                };
            }
            if no_change {
                n_scans = scan + 1;
                break;
            }
        }
        let value = vilb.expected_loss_unnormalized();
        if value < global_minimum {
            global_minimum = value;
            global_best = partition;
            global_n_scans = n_scans;
        }
    }
    // Canonicalize the labels
    global_best.canonicalize();
    let labels = global_best.labels_via_copying();
    let loss = (global_minimum + VarOfInfoLBComputer::expected_loss_constant(psm)) / (ni as f64);
    ((labels, loss, global_n_scans), candidates)
}

pub fn minimize_by_salso_slow(
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
            salso_engine_slow(f, g, psm, candidates, max_label, max_scans),
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
                    tx.send(salso_engine_slow(
                        f, g, psm, candidates, max_label, max_scans,
                    ))
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

pub fn salso_engine_slow(
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
        Partition::from(&global_best[..]).labels_via_copying(),
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
        0 => minimize_by_salso_slow(
            binder_single_partial,
            binder_single,
            max_size,
            &psm,
            candidates,
            max_scans,
            parallel,
        ),
        1 => minimize_by_salso_slow(
            vilb_single_partial,
            vilb_single,
            max_size,
            &psm,
            candidates,
            max_scans,
            parallel,
        ),
        2 => minimize_binder_by_salso(max_size, &psm, candidates, max_scans, parallel),
        3 => minimize_vilb_by_salso(max_size, &psm, candidates, max_scans, parallel),
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
