use crate::*;

pub fn expected_pairwise_allocation_matrix(partitions: &Vec<Partition>) -> Vec<f64> {
    let n_samples = partitions.len();
    assert!(
        n_samples > 0,
        "Number of partitions must be greater than 0.",
    );
    let n_items = partitions[0].n_items();
    assert!(
        partitions.iter().all(|x| n_items == x.n_items()),
        "All partitions must be of the same length."
    );
    let mut labels = vec![0usize; n_samples * n_items];
    for (i, partition) in partitions.iter().enumerate() {
        for (j, value) in partition.labels().iter().enumerate() {
            labels[n_samples * j + i] = *value;
        }
    }
    let mut counts = vec![0.0; n_items * n_items];
    expected_pairwise_allocation_matrix_engine(n_samples, n_items, &labels[..], &mut counts[..]);
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expected_pairwise_allocation_matrix() {
        let mut partitions = Vec::new();
        partitions.push(Partition::from("AABB".as_bytes()));
        partitions.push(Partition::from("AAAB".as_bytes()));
        partitions.push(Partition::from("ABBB".as_bytes()));
        partitions.push(Partition::from("AAAB".as_bytes()));
        let epam = expected_pairwise_allocation_matrix(&partitions);
        assert_eq!(format!("{:?}", epam), "[1.0, 0.75, 0.5, 0.0, 0.75, 1.0, 0.75, 0.25, 0.5, 0.75, 1.0, 0.5, 0.0, 0.25, 0.5, 1.0]");
    }

}

pub fn expected_pairwise_allocation_matrix_engine<A>(
    n_samples: usize,
    n_items: usize,
    partitions: &[A],
    counts: &mut [f64],
) -> ()
where
    A: PartialEq,
{
    let ns = n_samples;
    let ni = n_items;
    let nsf = ns as f64;
    for j in 0..ni {
        let nsj = ns * j;
        for k in 0..j {
            let nsk = ns * k;
            let mut count = 0usize;
            for i in 0..ns {
                unsafe {
                    if partitions.get_unchecked(nsj + i) == partitions.get_unchecked(nsk + i) {
                        count += 1;
                    }
                }
            }
            let proportion = count as f64 / nsf;
            unsafe {
                *counts.get_unchecked_mut(ni * j + k) = proportion;
                *counts.get_unchecked_mut(ni * k + j) = proportion;
            }
        }
        unsafe {
            *counts.get_unchecked_mut(ni * j + j) = 1.0;
        }
    }
}

use std::os::raw::{c_double, c_int};
use std::slice;

#[no_mangle]
pub unsafe extern "C" fn epam(
    n_samples: c_int,
    n_items: c_int,
    partitions_ptr: *const c_int,
    counts_ptr: *mut c_double,
) -> () {
    let ns = n_samples as usize;
    let ni = n_items as usize;
    let partitions: &[c_int] = slice::from_raw_parts(partitions_ptr, ns * ni);
    let counts: &mut [c_double] = slice::from_raw_parts_mut(counts_ptr, ni * ni);
    expected_pairwise_allocation_matrix_engine(ns, ni, partitions, counts);
}
