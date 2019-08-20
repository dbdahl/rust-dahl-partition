use crate::structure::*;
use crate::utils::bell;

use num_traits::cast::ToPrimitive;
use std::convert::TryFrom;

pub fn enumerate(n_items: usize) -> PartitionsHolder {
    let n_partitions = bell(n_items).to_usize().unwrap();
    let mut ph = PartitionsHolder::with_capacity(n_partitions, n_items);
    for partition in Partition::iter(n_items) {
        ph.push_slice(&partition[..]);
    }
    ph
}

#[no_mangle]
pub unsafe extern "C" fn dahl_partition__utils__enumerate(
    n_partitions: i32,
    n_items: i32,
    partitions_ptr: *mut i32,
) {
    let n_partitions = usize::try_from(n_partitions).unwrap();
    let n_items = usize::try_from(n_items).unwrap();
    let mut phv = PartitionsHolderView::from_ptr(partitions_ptr, n_partitions, n_items, true);
    for partition in Partition::iter(n_items) {
        phv.push_slice(&partition[..]);
    }
}
