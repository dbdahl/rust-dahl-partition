extern crate num_cpus;

use crate::*;
use std::os::raw::{c_double, c_int};
use std::slice;

pub fn psm<A>(partitions: &PartitionsHolderView<A>, parallel: bool) -> PairwiseSimilarityMatrix
where
    A: PartialEq + Sync + Send,
{
    let mut psm = PairwiseSimilarityMatrix::new(partitions.n_items());
    engine(
        partitions.n_samples(),
        partitions.n_items(),
        parallel,
        partitions,
        &mut psm.view(),
    );
    psm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psm() {
        let mut partitions = PartitionsHolder::new(4);
        partitions.push(Partition::from("AABB".as_bytes()));
        partitions.push(Partition::from("AAAB".as_bytes()));
        partitions.push(Partition::from("ABBB".as_bytes()));
        partitions.push(Partition::from("AAAB".as_bytes()));
        let partitions_view = partitions.view();
        let mut psm1 = psm(&partitions_view, true);
        assert_eq!(format!("{:?}", psm1.data), "[1.0, 0.75, 0.5, 0.0, 0.75, 1.0, 0.75, 0.25, 0.5, 0.75, 1.0, 0.5, 0.0, 0.25, 0.5, 1.0]");
        let mut psm2 = psm(&partitions_view, false);
        assert_eq!(format!("{:?}", psm2.data), "[1.0, 0.75, 0.5, 0.0, 0.75, 1.0, 0.75, 0.25, 0.5, 0.75, 1.0, 0.5, 0.0, 0.25, 0.5, 1.0]");
    }

}

fn engine<A>(
    n_samples: usize,
    n_items: usize,
    parallel: bool,
    partitions: &PartitionsHolderView<A>,
    psm: &mut PairwiseSimilarityMatrixView,
) -> ()
where
    A: PartialEq + Sync + Send,
{
    if !parallel {
        engine2(n_samples, n_items, None, partitions, psm);
    } else {
        let n_cores = num_cpus::get();
        let n_pairs = n_items * (n_items - 1) / 2;
        let step_size = n_pairs / n_cores + 1;
        let mut s = 0usize;
        let mut plan = Vec::with_capacity(n_cores + 1);
        plan.push(0);
        for i in 0..n_items {
            if s > step_size {
                plan.push(i);
                s = 0;
            }
            s += i;
        }
        while plan.len() < n_cores + 1 {
            plan.push(n_items);
        }
        crossbeam::scope(|s| {
            for i in 0..n_cores {
                let ptr =
                    unsafe { slice::from_raw_parts_mut(psm.data.as_mut_ptr(), psm.data.len()) };
                let lower = plan[i];
                let upper = plan[i + 1];
                s.spawn(move |_| {
                    let psm2 = &mut PairwiseSimilarityMatrixView::from_slice(ptr, n_items);
                    engine2(n_samples, n_items, Some(lower..upper), partitions, psm2);
                });
            }
        })
        .unwrap();
    }
}

fn engine2<A>(
    n_samples: usize,
    n_items: usize,
    range: Option<std::ops::Range<usize>>,
    partitions: &PartitionsHolderView<A>,
    psm: &mut PairwiseSimilarityMatrixView,
) -> ()
where
    A: PartialEq,
{
    let nsf = n_samples as f64;
    let indices = range.unwrap_or(0..n_items);
    for j in indices {
        for i in 0..j {
            let mut count = 0usize;
            for k in 0..n_samples {
                unsafe {
                    if partitions.get_unchecked((k, i)) == partitions.get_unchecked((k, j)) {
                        count += 1;
                    }
                }
            }
            let proportion = count as f64 / nsf;
            unsafe {
                *psm.get_unchecked_mut((i, j)) = proportion;
                *psm.get_unchecked_mut((j, i)) = proportion;
            }
        }
        unsafe {
            *psm.get_unchecked_mut((j, j)) = 1.0;
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dahl_partition__summary__psm(
    n_samples: c_int,
    n_items: c_int,
    parallel: c_int,
    partitions_ptr: *const c_int,
    psm_ptr: *mut c_double,
) -> () {
    let ns = n_samples as usize;
    let ni = n_items as usize;
    let partitions = PartitionsHolderView::from_ptr(partitions_ptr, ns, ni, true);
    let mut psm = PairwiseSimilarityMatrixView::from_ptr(psm_ptr, ni);
    engine(ns, ni, parallel != 0, &partitions, &mut psm);
}
