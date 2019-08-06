use crate::structure::*;

pub fn enumerate<'a>(n_items: usize, phv: &mut PartitionsHolderView<'a>) {
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
