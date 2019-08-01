use crate::*;
use std::os::raw::{c_double, c_int};
use std::slice;

pub fn expected_pairwise_allocation_matrix(
    partitions: &Vec<Partition>,
    parallel: bool,
) -> Vec<f64> {
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
    expected_pairwise_allocation_matrix_engine(
        n_samples,
        n_items,
        parallel,
        &labels[..],
        &mut counts[..],
    );
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
        let epam = expected_pairwise_allocation_matrix(&partitions, true);
        assert_eq!(format!("{:?}", epam), "[1.0, 0.75, 0.5, 0.0, 0.75, 1.0, 0.75, 0.25, 0.5, 0.75, 1.0, 0.5, 0.0, 0.25, 0.5, 1.0]");
    }

}

pub fn expected_pairwise_allocation_matrix_engine<A>(
    n_samples: usize,
    n_items: usize,
    parallel: bool,
    partitions: &[A],
    counts: &mut [f64],
) -> ()
where
    A: PartialEq + Sync + Send,
{
    if !parallel {
        expected_pairwise_allocation_matrix_engine2(n_samples, n_items, None, partitions, counts);
    } else {
        println!("************** Parallel");
        let schedule: Vec<usize> = (0..n_items).collect();
        crossbeam::scope(|s| {
            for i in schedule {
                let counts2 =
                    unsafe { slice::from_raw_parts_mut(counts.as_mut_ptr(), counts.len()) };
                s.spawn(move |_| {
                    expected_pairwise_allocation_matrix_engine2(
                        n_samples,
                        n_items,
                        Some(i..(i + 1)),
                        partitions,
                        counts2,
                    );
                });
            }
        })
        .unwrap();
        /*
        let mut handles = Vec::with_capacity(n_items);
        for i in 0..n_items {
            handles[i] = {
                thread::spawn(|| {
                    expected_pairwise_allocation_matrix_engine2(
                        n_samples,
                        n_items,
                        Some(0..(n_items / 2)),
                        partitions,
                        counts,
                    );
                })
            }
        }
        handles.iter_mut().for_each(|handle| handle.join().unwrap());
        */
    }
}

pub fn expected_pairwise_allocation_matrix_engine2<A>(
    n_samples: usize,
    n_items: usize,
    range: Option<std::ops::Range<usize>>,
    partitions: &[A],
    counts: &mut [f64],
) -> ()
where
    A: PartialEq,
{
    let nsf = n_samples as f64;
    let indices = range.unwrap_or(0..n_items);
    for j in indices {
        let nsj = n_samples * j;
        for k in 0..j {
            let nsk = n_samples * k;
            let mut count = 0usize;
            for i in 0..n_samples {
                unsafe {
                    if partitions.get_unchecked(nsj + i) == partitions.get_unchecked(nsk + i) {
                        count += 1;
                    }
                }
            }
            let proportion = count as f64 / nsf;
            unsafe {
                *counts.get_unchecked_mut(n_items * j + k) = proportion;
                *counts.get_unchecked_mut(n_items * k + j) = proportion;
            }
        }
        unsafe {
            *counts.get_unchecked_mut(n_items * j + j) = 1.0;
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_partition__summary__epam__expected_pairwise_allocation_matrix(
    n_samples: c_int,
    n_items: c_int,
    parallel: c_int,
    partitions_ptr: *const c_int,
    counts_ptr: *mut c_double,
) -> () {
    let ns = n_samples as usize;
    let ni = n_items as usize;
    let partitions: &[c_int] = slice::from_raw_parts(partitions_ptr, ns * ni);
    let counts: &mut [c_double] = slice::from_raw_parts_mut(counts_ptr, ni * ni);
    expected_pairwise_allocation_matrix_engine(ns, ni, parallel != 0, partitions, counts);
}
