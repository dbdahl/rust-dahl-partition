use crate::structure::*;

pub fn enumerate(phv: &mut PartitionsHolderView) {
    let n_partitions = phv.n_partitions();
    let n_items = phv.n_items();
    let mut state = vec![0; n_items];
    let mut max = vec![0; n_items];
    let mut i = n_items;
    if i == n_items {
        let partition = Partition::from(&state[..]);
        phv.push(&partition);
        i -= 1;
        if state[i] < max[i] {
            state[i] += 1;
            i += 1;
        } else {
            state[i] = 0;
            i -= 1;
        }
    }
}

struct PartitionLabels {
    count: u32,
}

impl PartitionLabels {
    fn new() -> PartitionLabels {
        PartitionLabels { count: 0 }
    }
}

impl Iterator for PartitionLabels {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        self.count += 1;
        if self.count < 6 {
            Some(self.count)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate() {
        let mut ph = PartitionsHolder::allocated(4, 3, true);
        let mut phv = ph.view();
        // enumerate(&mut phv);
        println!("{}", phv.to_string());
        let mut c = PartitionLabels::new();
        //for i in c {
        //    println!("{}", i);
        //}
        //assert_eq!(phv.to_string(), "1");
    }

}
