extern crate rand;

use crate::Partition;
use rand::distributions::{Distribution, WeightedIndex};

pub fn sample(n_items: usize, mass: f64) -> Partition {
    assert!(mass > 0.0, "Mass must be greater than 0.0.");
    let mut rng = rand::thread_rng();
    let mut p = Partition::new(n_items);
    for i in 0..p.n_items() {
        match p.subsets().last() {
            None => p.new_subset(),
            Some(last) => {
                if !last.is_empty() {
                    p.new_subset()
                }
            }
        }
        let probs = p.subsets().iter().map(|subset| {
            if subset.is_empty() {
                mass
            } else {
                subset.n_items() as f64
            }
        });
        let dist = WeightedIndex::new(probs).unwrap();
        let subset_index = dist.sample(&mut rng);
        p.add_with_index(i, subset_index);
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::summary::epam::expected_pairwise_allocation_matrix;

    #[test]
    fn test_sample() {
        let n_samples = 10000;
        let n_items = 4;
        let mass = 2.0;
        let mut samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            samples.push(sample(n_items, mass));
        }
        let epam = expected_pairwise_allocation_matrix(&samples, true);
        let truth = 1.0 / (1.0 + mass);
        let margin_of_error = 3.58 * (truth * (1.0 - truth) / n_samples as f64).sqrt();
        assert!(epam.iter().all(|prob| {
            *prob == 1.0 || (truth - margin_of_error < *prob && *prob < truth + margin_of_error)
        }));
    }

}

use std::convert::TryFrom;
use std::os::raw::{c_double, c_int};
use std::slice;

#[no_mangle]
pub unsafe extern "C" fn dahl_partition__distribution__crp__sample(
    n_samples: c_int,
    n_items: c_int,
    mass: c_double,
    ptr: *mut c_int,
) -> () {
    let ns = n_samples as usize;
    let ni = n_items as usize;
    let array: &mut [c_int] = slice::from_raw_parts_mut(ptr, ns * ni);
    for i in 0..ns {
        let p = sample(ni, mass);
        let labels = p.labels_with_missing();
        for j in 0..ni {
            array[ns * j + i] = c_int::try_from(labels[j].unwrap()).unwrap();
        }
    }
}
