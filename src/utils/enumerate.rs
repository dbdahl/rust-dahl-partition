use crate::structure::*;

pub fn enumerate(n_items: usize, phv: &mut PartitionsHolderView) {
    let n_partitions = phv.n_partitions();
    for i in 0..n_partitions {
        let partition = if i % 2 == 0 {
            Partition::in_one_subset(n_items)
        } else {
            Partition::in_one_subset(n_items)
        };
        phv.push(&partition);
    }
}
