use crate::structure::*;

pub fn enumerate(phv: &mut PartitionsHolderView) {
    let n_partitions = phv.n_partitions();
    let n_items = phv.n_items();
    for i in 0..n_partitions {
        let partition = if i % 2 == 0 {
            Partition::one_subset(n_items)
        } else {
            Partition::singleton_subsets(n_items)
        };
        phv.push(&partition);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate() {
        let mut ph = PartitionsHolder::allocated(4, 2, true);
        let mut phv = ph.view();
        //enumerate(&mut phv);
        //assert_eq!(phv.to_string(), "1");
    }

}
