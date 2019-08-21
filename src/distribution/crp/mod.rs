extern crate rand;

use crate::structure::*;

use rand::distributions::{Distribution, WeightedIndex};
use std::convert::TryFrom;
use std::slice;

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

    #[test]
    fn test_sample() {
        let n_partitions = 10000;
        let n_items = 4;
        let mass = 2.0;
        let mut samples = PartitionsHolder::with_capacity(n_partitions, n_items);
        for _ in 0..n_partitions {
            samples.push_partition(&sample(n_items, mass));
        }
        let mut psm = crate::summary::psm::psm(&samples.view(), true);
        let truth = 1.0 / (1.0 + mass);
        let margin_of_error = 3.58 * (truth * (1.0 - truth) / n_partitions as f64).sqrt();
        assert!(psm.view().data().iter().all(|prob| {
            *prob == 1.0 || (truth - margin_of_error < *prob && *prob < truth + margin_of_error)
        }));
    }

}

#[no_mangle]
pub unsafe extern "C" fn dahl_partition__distribution__crp__sample(
    n_partitions: i32,
    n_items: i32,
    mass: f64,
    ptr: *mut i32,
) -> () {
    let np = n_partitions as usize;
    let ni = n_items as usize;
    let array: &mut [i32] = slice::from_raw_parts_mut(ptr, np * ni);
    for i in 0..np {
        let p = sample(ni, mass);
        let labels = p.labels();
        for j in 0..ni {
            array[np * j + i] = i32::try_from(labels[j].unwrap()).unwrap();
        }
    }
}
