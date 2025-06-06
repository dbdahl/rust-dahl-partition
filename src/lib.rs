#![allow(dead_code)]

extern crate rand;

use rand::seq::SliceRandom;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::HashSet;
use std::convert::TryFrom;
use std::fmt;
use std::hash::Hash;
use std::slice;

/// A partition of `n_items` is a set of subsets such that the subsets are mutually exclusive,
/// nonempty, and exhaustive.  This data structure enforces mutually exclusivity and provides
/// methods to test for the other two properties:
/// [subsets_are_nonempty](struct.Partition.html#method.subsets_are_nonempty) and
/// [subsets_are_exhaustive](struct.Partition.html#method.subsets_are_exhaustive).
/// Zero-based numbering is used, e.g., the first
/// item is index 0 and the first subset is subset 0.  When a subset is added to the data structure,
/// the next available integer label is used for its subset index.  To remove empty subsets and
/// relabel subsets into the canonical form, use the
/// [canonicalize](struct.Partition.html#method.canonicalize) method.
///
#[derive(Debug, Clone)]
pub struct Partition {
    n_items: usize,
    n_allocated_items: usize,
    subsets: Vec<Subset>,
    labels: Vec<Option<usize>>,
}

impl Partition {
    /// Instantiates a partition for `n_items`, none of which are initially allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use dahl_partition::*;
    /// let partition = Partition::new(10);
    /// assert_eq!(partition.n_items(), 10);
    /// assert_eq!(partition.n_subsets(), 0);
    /// assert_eq!(partition.to_string(), "_ _ _ _ _ _ _ _ _ _");
    /// ```
    pub fn new(n_items: usize) -> Self {
        Self {
            n_items,
            n_allocated_items: 0,
            subsets: Vec::new(),
            labels: vec![None; n_items],
        }
    }

    /// Instantiates a partition for `n_items`, with all items allocated to one subset.
    ///
    pub fn one_subset(n_items: usize) -> Self {
        let mut subset = Subset::new();
        for i in 0..n_items {
            subset.add(i);
        }
        let labels = vec![Some(0); n_items];
        Self {
            n_items,
            n_allocated_items: n_items,
            subsets: vec![subset],
            labels,
        }
    }

    /// Instantiates a partition for `n_items`, with each item allocated to its own subset.
    ///
    pub fn singleton_subsets(n_items: usize) -> Self {
        let mut subsets = Vec::with_capacity(n_items);
        let mut labels = Vec::with_capacity(n_items);
        for i in 0..n_items {
            let mut subset = Subset::new();
            subset.add(i);
            subsets.push(subset);
            labels.push(Some(i));
        }
        Self {
            n_items,
            n_allocated_items: n_items,
            subsets,
            labels,
        }
    }

    /// Instantiates a partition from a slice of subset labels.
    /// Items `i` and `j` are in the subset if and only if their labels are equal.
    ///
    /// # Examples
    ///
    /// ```
    /// use dahl_partition::*;
    /// let labels = vec!('a', 'a', 'a', 't', 't', 'a', 'a', 'w', 'a', 'w');
    /// let partition1 = Partition::from(&labels[..]);
    /// let partition2 = Partition::from(&[2,2,2,5,5,2,2,7,2,7]);
    /// let partition3 = Partition::from("AAABBAACAC".as_bytes());
    /// assert_eq!(partition1, partition2);
    /// assert_eq!(partition2, partition3);
    /// assert_eq!(partition1.n_items(), labels.len());
    /// assert_eq!(partition1.n_subsets(), {
    ///     let mut x = labels.clone();
    ///     x.sort();
    ///     x.dedup();
    ///     x.len()
    /// });
    /// assert_eq!(partition1.to_string(), "0 0 0 1 1 0 0 2 0 2");
    /// ```
    pub fn from<T>(labels: &[T]) -> Self
    where
        T: Eq + Hash,
    {
        let n_items = labels.len();
        let mut new_labels = vec![None; n_items];
        let mut subsets = Vec::new();
        let mut map = HashMap::new();
        let mut next_label: usize = 0;
        for i in 0..labels.len() {
            let key = &labels[i];
            let label = map.entry(key).or_insert_with(|| {
                subsets.push(Subset::new());
                let label = next_label;
                next_label += 1;
                label
            });
            subsets[*label].add(i);
            new_labels[i] = Some(*label);
        }
        Self {
            n_items,
            n_allocated_items: n_items,
            subsets,
            labels: new_labels,
        }
    }

    /// An iterator over all possible partitions for the specified number of items.
    ///
    pub fn iter(n_items: usize) -> PartitionIterator {
        PartitionIterator {
            n_items,
            labels: vec![0; n_items],
            max: vec![0; n_items],
            done: false,
            period: 1,
        }
    }

    /// A vector of iterators, which together go over all possible partitions for the specified
    /// number of items.  This can be useful in multi-thread situations.
    ///
    pub fn iter_sharded(n_shards: u32, n_items: usize) -> Vec<PartitionIterator> {
        let mut shards = Vec::with_capacity(n_shards as usize);
        let ns = if n_shards == 0 { 1 } else { n_shards };
        for i in 0..ns {
            let mut iter = PartitionIterator {
                n_items,
                labels: vec![0; n_items],
                max: vec![0; n_items],
                done: false,
                period: ns,
            };
            iter.advance(i);
            shards.push(iter);
        }
        shards
    }

    /// Number of items that can be allocated to the partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use dahl_partition::*;
    /// let partition = Partition::from("AAABBAACAC".as_bytes());
    /// assert_eq!(partition.n_items(), 10);
    /// ```
    pub fn n_items(&self) -> usize {
        self.n_items
    }

    /// Number of subsets in the partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use dahl_partition::*;
    /// let mut partition = Partition::from("AAABBAACAC".as_bytes());
    /// assert_eq!(partition.n_subsets(), 3);
    /// partition.remove(7);
    /// partition.remove(9);
    /// assert_eq!(partition.n_subsets(), 3);
    /// partition.canonicalize();
    /// assert_eq!(partition.n_subsets(), 2);
    /// ```
    pub fn n_subsets(&self) -> usize {
        self.subsets.len()
    }

    /// A vector of length `n_items` whose elements are subset labels.  Panics if any items are
    /// not allocated.
    pub fn labels_via_copying(&self) -> Vec<usize> {
        self.labels.iter().map(|x| x.unwrap()).collect()
    }

    /// Copy subset labels into a slice.  Panics if any items are not allocated or if the slice
    /// is not the correct length.
    pub fn labels_into_slice<T, U>(&self, slice: &mut [T], f: U)
    where
        U: Fn(&Option<usize>) -> T,
    {
        for (x, y) in slice.iter_mut().zip(self.labels.iter().map(f)) {
            *x = y;
        }
    }

    /// A reference to a vector of length `n_items` whose elements are `None` for items that are
    /// not allocated (i.e., missing) and, for items that are allocated, `Some(subset_index)` where `subset_index`
    /// is the index of the subset to which the item is allocated.
    pub fn labels(&self) -> &Vec<Option<usize>> {
        &self.labels
    }

    /// Either `None` for an item that is
    /// not allocated (i.e., missing) or, for an item that is allocated, `Some(subset_index)` where `subset_index`
    /// is the index of the subset to which the item is allocated.
    pub fn label_of(&self, item_index: usize) -> Option<usize> {
        self.check_item_index(item_index);
        self.labels[item_index]
    }

    /// Either `None` for an item that is not allocated (i.e., missing) or, for an item that is allocated,
    /// `Some(&subset)` where `&subset` is a reference to the subset to which the item is allocated.
    pub fn subset_of(&self, item_index: usize) -> Option<&Subset> {
        self.label_of(item_index)
            .map(|subset_index| &self.subsets[subset_index])
    }

    /// Returns `true` if and only if both items are allocated (i.e., not missing) and are allocated
    /// to the same subset.
    pub fn paired(&self, item1_index: usize, item2_index: usize) -> bool {
        self.check_item_index(item1_index);
        self.check_item_index(item2_index);
        let l1 = self.labels[item1_index];
        l1.is_some() && l1 == self.labels[item2_index]
    }

    pub fn subsets(&self) -> &Vec<Subset> {
        &self.subsets
    }

    pub fn clean_subset(&mut self, subset_index: usize) {
        self.check_subset_index(subset_index);
        self.subsets[subset_index].clean();
    }

    /// Add a new empty subset to the partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use dahl_partition::*;
    /// let mut partition = Partition::from("AAABBAACAC".as_bytes());
    /// assert_eq!(partition.n_subsets(), 3);
    /// assert!(partition.subsets_are_nonempty());
    /// partition.new_subset();
    /// assert!(!partition.subsets_are_nonempty());
    /// assert_eq!(partition.n_subsets(), 4);
    /// ```
    pub fn new_subset(&mut self) {
        self.subsets.push(Subset::new());
    }

    /// Add a new subset containing `item_index` to the partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use dahl_partition::*;
    /// let mut partition = Partition::new(3);
    /// partition.add(1);
    /// assert_eq!(partition.to_string(), "_ 0 _");
    /// partition.add(0);
    /// assert_eq!(partition.to_string(), "1 0 _");
    /// partition.canonicalize();
    /// assert_eq!(partition.to_string(), "0 1 _");
    /// ```
    pub fn add(&mut self, item_index: usize) -> &mut Self {
        self.check_item_index(item_index);
        self.check_not_allocated(item_index);
        self.n_allocated_items += 1;
        self.subsets.push(Subset::new());
        self.add_engine(item_index, self.subsets.len() - 1);
        self
    }

    /// Add item `item_index` to the subset `subset_index` of the partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use dahl_partition::*;
    /// let mut partition = Partition::new(3);
    /// partition.add(1);
    /// assert_eq!(partition.to_string(), "_ 0 _");
    /// partition.add_with_index(2, 0);
    /// assert_eq!(partition.to_string(), "_ 0 0");
    /// ```
    pub fn add_with_index(&mut self, item_index: usize, subset_index: usize) -> &mut Self {
        self.check_item_index(item_index);
        self.check_not_allocated(item_index);
        self.check_subset_index(subset_index);
        self.n_allocated_items += 1;
        self.add_engine(item_index, subset_index);
        self
    }

    /// Remove item `item_index` from the partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use dahl_partition::*;
    /// let mut partition = Partition::from("AAABBAACAC".as_bytes());
    /// partition.remove(1);
    /// partition.remove(4);
    /// assert_eq!(partition.to_string(), "0 _ 0 1 _ 0 0 2 0 2");
    /// partition.remove(3);
    /// assert_eq!(partition.to_string(), "0 _ 0 _ _ 0 0 2 0 2");
    /// partition.canonicalize();
    /// assert_eq!(partition.to_string(), "0 _ 0 _ _ 0 0 1 0 1");
    /// ```
    pub fn remove(&mut self, item_index: usize) -> &mut Self {
        self.check_item_index(item_index);
        self.remove_engine(item_index, self.check_allocated(item_index));
        self
    }

    /// Remove item `item_index` from the partition and, if this creates an empty subset, relabel.
    ///
    /// # Examples
    ///
    /// ```
    /// use dahl_partition::*;
    /// let mut partition = Partition::from("AAABBAACAC".as_bytes());
    /// partition.remove_clean_and_relabel(1, |x,y| {});
    /// partition.remove_clean_and_relabel(4, |x,y| {});
    /// partition.remove_clean_and_relabel(3, |x,y| {});
    /// assert_eq!(partition.to_string(), "0 _ 0 _ _ 0 0 1 0 1");
    /// ```
    pub fn remove_clean_and_relabel<T>(&mut self, item_index: usize, mut callback: T) -> &mut Self
    where
        T: FnMut(usize, usize),
    {
        self.check_item_index(item_index);
        let subset_index = self.check_allocated(item_index);
        self.remove_engine(item_index, subset_index);
        if self.subsets[subset_index].is_empty() {
            let moved_subset_index = self.subsets.len() - 1;
            if moved_subset_index != subset_index {
                for i in self.subsets[moved_subset_index].items() {
                    self.labels[*i] = Some(subset_index);
                }
            }
            callback(subset_index, moved_subset_index);
            self.clean_subset(subset_index);
            self.subsets.swap_remove(subset_index);
        } else {
            self.subsets[subset_index].clean();
        }
        self
    }

    /// Remove item `item_index` from the subset `subset_index` of the partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use dahl_partition::*;
    /// let mut partition = Partition::from("AAABBAACAC".as_bytes());
    /// partition.remove_with_index(1,0);
    /// partition.remove_with_index(4,1);
    /// assert_eq!(partition.to_string(), "0 _ 0 1 _ 0 0 2 0 2");
    /// partition.remove_with_index(3,1);
    /// assert_eq!(partition.to_string(), "0 _ 0 _ _ 0 0 2 0 2");
    /// partition.canonicalize();
    /// assert_eq!(partition.to_string(), "0 _ 0 _ _ 0 0 1 0 1");
    /// ```
    pub fn remove_with_index(&mut self, item_index: usize, subset_index: usize) -> &mut Self {
        self.check_item_index(item_index);
        self.check_item_in_subset(item_index, subset_index);
        self.remove_engine(item_index, subset_index);
        self
    }

    /// Removes that last item from the subset `subset_index` of the partition.  This may or may not
    /// be the most recently added item.
    ///
    /// # Examples
    ///
    /// ```
    /// use dahl_partition::*;
    /// let mut partition = Partition::from("AAABBAACAC".as_bytes());
    /// assert_eq!(partition.pop_item(1),Some(4));
    /// ```
    pub fn pop_item(&mut self, subset_index: usize) -> Option<usize> {
        self.check_subset_index(subset_index);
        let i_option = self.subsets[subset_index].pop();
        if let Some(item_index) = i_option {
            self.labels[item_index] = None;
            self.n_allocated_items -= 1;
        }
        i_option
    }

    /// Removes that last subset of the partition if it empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use dahl_partition::*;
    /// let mut partition = Partition::from("AAABBAACAC".as_bytes());
    /// partition.new_subset();
    /// assert_ne!(partition.pop_subset(),None);
    /// ```
    pub fn pop_subset(&mut self) -> Option<Subset> {
        if self.subsets.is_empty() {
            return None;
        }
        let subset = self.subsets.last().unwrap();
        if !subset.is_empty() {
            return None;
        }
        self.subsets.pop()
    }

    /// Transfers item `item_index` to a new empty subset.
    pub fn transfer(&mut self, item_index: usize) -> &mut Self {
        self.check_item_index(item_index);
        let subset_index = self.check_allocated(item_index);
        self.subsets[subset_index].remove(item_index);
        self.subsets.push(Subset::new());
        self.add_engine(item_index, self.subsets.len() - 1);
        self
    }

    /// Transfers item `item_index` from the subset `old_subset_index` to another subset
    /// `new_subset_index`.
    pub fn transfer_with_index(
        &mut self,
        item_index: usize,
        old_subset_index: usize,
        new_subset_index: usize,
    ) -> &mut Self {
        self.check_item_index(item_index);
        self.check_item_in_subset(item_index, old_subset_index);
        self.check_subset_index(new_subset_index);
        self.subsets[old_subset_index].remove(item_index);
        self.add_engine(item_index, new_subset_index);
        self
    }

    /// Put partition into canonical form, removing empty subsets and consecutively numbering
    /// subsets starting at 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use dahl_partition::*;
    /// let mut partition = Partition::new(3);
    /// partition.add(1);
    /// partition.add(2);
    /// partition.add(0);
    /// assert_eq!(partition.to_string(), "2 0 1");
    /// partition.canonicalize();
    /// assert_eq!(partition.to_string(), "0 1 2");
    /// ```
    ///
    pub fn canonicalize(&mut self) -> &mut Self {
        self.canonicalize_by_permutation(None)
    }

    /// Put partition into canonical form according to a permutation, removing empty subsets and consecutively numbering
    /// subsets starting at 0 based on the supplied permutation.  If `None` is supplied, the natural permutation is used.
    ///
    /// # Examples
    ///
    /// ```
    /// use dahl_partition::*;
    /// use std::fs::canonicalize;
    /// let mut partition = Partition::new(3);
    /// partition.add(1);
    /// partition.add(2);
    /// partition.add(0);
    /// assert_eq!(partition.to_string(), "2 0 1");
    /// partition.canonicalize_by_permutation(Some(&Permutation::from_slice(&[0,2,1]).unwrap()));
    /// assert_eq!(partition.to_string(), "0 2 1");
    /// ```
    ///
    pub fn canonicalize_by_permutation(&mut self, permutation: Option<&Permutation>) -> &mut Self {
        if let Some(p) = permutation {
            assert_eq!(self.n_items, p.len());
        };
        if self.n_allocated_items == 0 {
            return self;
        }
        let mut new_labels = vec![None; self.n_items];
        let mut next_label: usize = 0;
        for i in 0..self.n_items {
            let ii = match permutation {
                None => i,
                Some(p) => p[i],
            };
            match new_labels[ii] {
                Some(_) => (),
                None => match self.labels[ii] {
                    None => (),
                    Some(subset_index) => {
                        let subset = &mut self.subsets[subset_index];
                        subset.clean();
                        let some_label = Some(next_label);
                        next_label += 1;
                        for j in subset.vector.iter() {
                            new_labels[*j] = some_label;
                        }
                    }
                },
            }
        }
        self.subsets.sort_unstable_by(|x, y| {
            if x.is_empty() {
                if y.is_empty() {
                    Ordering::Equal
                } else {
                    Ordering::Greater
                }
            } else if y.is_empty() {
                Ordering::Less
            } else {
                new_labels[x.vector[0]]
                    .unwrap()
                    .cmp(&new_labels[y.vector[0]].unwrap())
            }
        });
        while !self.subsets.is_empty() && self.subsets.last().unwrap().is_empty() {
            self.subsets.pop();
        }
        self.labels = new_labels;
        self
    }

    /// Test whether subsets are exhaustive.
    pub fn subsets_are_exhaustive(&self) -> bool {
        self.n_allocated_items == self.n_items
    }

    /// Test whether subsets are nonempty.
    pub fn subsets_are_nonempty(&self) -> bool {
        self.subsets.iter().all(|x| !x.is_empty())
    }

    fn add_engine(&mut self, item_index: usize, subset_index: usize) {
        self.labels[item_index] = Some(subset_index);
        self.subsets[subset_index].add(item_index);
    }

    fn remove_engine(&mut self, item_index: usize, subset_index: usize) {
        self.labels[item_index] = None;
        self.subsets[subset_index].remove(item_index);
        self.n_allocated_items -= 1;
    }

    fn check_item_index(&self, item_index: usize) {
        if item_index >= self.n_items {
            panic!(
                "Attempt to allocate item {} when only {} are available.",
                item_index, self.n_items
            );
        };
    }

    fn check_subset_index(&self, subset_index: usize) {
        if subset_index >= self.n_subsets() {
            panic!(
                "Attempt to allocate to subset {} when only {} are available.",
                subset_index,
                self.n_subsets()
            );
        };
    }

    fn check_allocated(&self, item_index: usize) -> usize {
        match self.labels[item_index] {
            Some(subset_index) => subset_index,
            None => panic!("Item {} is not allocated.", item_index),
        }
    }

    fn check_not_allocated(&self, item_index: usize) {
        if let Some(j) = self.labels[item_index] {
            panic!("Item {} is already allocated to subset {}.", item_index, j)
        };
    }

    fn check_item_in_subset(&self, item_index: usize, subset_index: usize) {
        match self.labels[item_index] {
            Some(j) => {
                if j != subset_index {
                    panic!("Item {} is already allocated to subset {}.", item_index, j);
                };
            }
            None => panic!("Item {} is not allocated to any subset.", item_index),
        };
    }
}

impl fmt::Display for Partition {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut str = "";
        for i in 0..self.n_items {
            fmt.write_str(str)?;
            str = " ";
            match self.labels[i] {
                None => fmt.write_str("_")?,
                Some(subset_index) => {
                    let x = subset_index.to_string();
                    fmt.write_str(&x[..])?;
                }
            };
        }
        Ok(())
    }
}

impl PartialEq for Partition {
    fn eq(&self, other: &Self) -> bool {
        self.n_items() == other.n_items
            && self.n_allocated_items == other.n_allocated_items
            && self.subsets.len() == other.subsets.len()
            && {
                let mut x = self.clone();
                let x1 = x.canonicalize();
                let mut y = other.clone();
                let y1 = y.canonicalize();
                x1.subsets == y1.subsets
            }
    }
}

#[cfg(test)]
mod tests_partition {
    use super::*;

    #[test]
    fn test_complete() {
        let mut p = Partition::new(6);
        p.add(2);
        for i in &[3, 4, 5] {
            p.add_with_index(*i, 0);
        }
        assert_eq!(p.to_string(), "_ _ 0 0 0 0");
        assert!(!p.subsets_are_exhaustive());
        p.add(1);
        p.add_with_index(0, 1);
        p.canonicalize();
        assert_eq!(p.to_string(), "0 0 1 1 1 1");
        p.remove_with_index(2, 1);
        p.canonicalize();
        assert_eq!(p.to_string(), "0 0 _ 1 1 1");
        assert!(!p.subsets_are_exhaustive());
        p.remove_with_index(0, 0);
        p.remove_with_index(1, 0);
        assert_eq!(p.to_string(), "_ _ _ 1 1 1");
        p.canonicalize();
        assert_eq!(p.to_string(), "_ _ _ 0 0 0");
        p.transfer(5);
        p.transfer_with_index(3, 0, 1);
        assert_eq!(p.to_string(), "_ _ _ 1 0 1");
        p.add_with_index(1, 1);
        assert_eq!(p.to_string(), "_ 1 _ 1 0 1");
        p.canonicalize();
        assert_eq!(p.to_string(), "_ 0 _ 0 1 0");
        assert_eq!(p.n_subsets(), 2);
        p.remove_with_index(4, 1);
        assert_eq!(p.n_subsets(), 2);
        p.canonicalize();
        assert_eq!(p.n_subsets(), 1);
    }

    #[test]
    fn test_canonicalize() {
        let mut p = Partition::new(6);
        p.add(0)
            .add_with_index(1, 0)
            .add(2)
            .add_with_index(3, 1)
            .add_with_index(4, 0);
        assert_eq!(p.to_string(), "0 0 1 1 0 _");
        p.add(5);
        assert_eq!(p.to_string(), "0 0 1 1 0 2");
        p.remove(2);
        p.remove(3);
        assert_eq!(p.to_string(), "0 0 _ _ 0 2");
        p.canonicalize();
        assert_eq!(p.to_string(), "0 0 _ _ 0 1");
    }

    #[test]
    fn test_from() {
        let p0 = Partition::from(&[45, 34, 23, 23, 23, 24, 45]);
        assert_eq!(p0.to_string(), "0 1 2 2 2 3 0");
        let p1 = Partition::from("ABCAADAABD".as_bytes());
        assert_eq!(p1.to_string(), "0 1 2 0 0 3 0 0 1 3");
        let p2 = Partition::from(p1.labels());
        assert_eq!(p1.to_string(), p2.to_string());
        let p3 = Partition::from(p0.labels());
        assert_eq!(p0.to_string(), p3.to_string());
    }

    #[test]
    fn test_equality() {
        let p0 = Partition::from("ABCAADAABD".as_bytes());
        let p1 = Partition::from("ABCAADAABD".as_bytes());
        let p2 = Partition::from("ABCRRRRRRD".as_bytes());
        assert_eq!(p0, p1);
        assert_ne!(p0, p2);
    }
}

#[doc(hidden)]
pub struct PartitionIterator {
    n_items: usize,
    labels: Vec<usize>,
    max: Vec<usize>,
    done: bool,
    period: u32,
}

impl PartitionIterator {
    fn advance(&mut self, times: u32) {
        for _ in 0..times {
            let mut i = self.n_items - 1;
            while (i > 0) && (self.labels[i] == self.max[i - 1] + 1) {
                self.labels[i] = 0;
                self.max[i] = self.max[i - 1];
                i -= 1;
            }
            if i == 0 {
                self.done = true;
                return;
            }
            self.labels[i] += 1;
            let m = self.max[i].max(self.labels[i]);
            self.max[i] = m;
            i += 1;
            while i < self.n_items {
                self.max[i] = m;
                self.labels[i] = 0;
                i += 1;
            }
        }
    }
}

impl Iterator for PartitionIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            None
        } else {
            let result = Some(self.labels.clone());
            self.advance(self.period);
            result
        }
    }
}

#[cfg(test)]
mod tests_iterator {
    use super::*;

    fn test_iterator_engine(n_items: usize, bell_number: usize) {
        let mut counter = 0usize;
        for _ in Partition::iter(n_items) {
            counter += 1;
        }
        assert_eq!(counter, bell_number);
    }

    #[test]
    fn test_iterator() {
        test_iterator_engine(1, 1);
        test_iterator_engine(2, 2);
        let mut ph = PartitionsHolder::with_capacity(5, 3);
        for labels in Partition::iter(3) {
            ph.push_slice(&labels[..]);
        }
        assert_eq!(
            ph.view().data(),
            &[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 2]
        );
        test_iterator_engine(10, 115_975);
    }

    #[test]
    fn test_sharded() {
        let mut ph = PartitionsHolder::new(3);
        for iter in Partition::iter_sharded(8, 3) {
            for labels in iter {
                ph.push_slice(&labels[..]);
            }
        }
        assert_eq!(
            ph.view().data(),
            &[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 2]
        );
    }
}

/// A data structure representing sets of integers.  The user instantiates subsets through
/// [Partition](struct.Partition.html).
///
#[derive(Debug, Clone)]
pub struct Subset {
    n_items: usize,
    set: HashSet<usize>,
    vector: Vec<usize>,
    is_clean: bool,
}

impl Subset {
    pub fn new() -> Subset {
        Subset {
            n_items: 0,
            set: HashSet::new(),
            vector: Vec::new(),
            is_clean: true,
        }
    }

    fn from<'a, I>(x: I) -> Subset
    where
        I: Iterator<Item = &'a usize>,
    {
        let mut set = HashSet::new();
        let mut vector = Vec::new();
        let mut n_items = 0;
        for i in x {
            if set.insert(*i) {
                n_items += 1;
                vector.push(*i);
            }
        }
        Subset {
            n_items,
            set,
            vector,
            is_clean: true,
        }
    }

    pub fn add(&mut self, i: usize) -> bool {
        if self.set.insert(i) {
            self.n_items += 1;
            if self.is_clean {
                self.vector.push(i);
            }
            true
        } else {
            false
        }
    }

    fn merge(&mut self, other: &Subset) -> bool {
        let mut added = false;
        if other.is_clean {
            for i in other.vector.iter() {
                added = self.add(*i) || added;
            }
        } else {
            for i in other.set.iter() {
                added = self.add(*i) || added;
            }
        }
        added
    }

    pub fn intersection_count(&self, other: &Subset) -> usize {
        if self.n_items > other.n_items {
            other.intersection_count(self)
        } else {
            let mut count = 0;
            if self.is_clean {
                for i in self.vector.iter() {
                    if other.contains(*i) {
                        count += 1;
                    };
                }
            } else if other.is_clean {
                for i in other.vector.iter() {
                    if self.contains(*i) {
                        count += 1;
                    };
                }
            } else {
                for i in self.set.iter() {
                    if other.contains(*i) {
                        count += 1;
                    };
                }
            };
            count
        }
    }

    pub fn intersection(&self, other: &Subset) -> Subset {
        let set: HashSet<usize> = self.set.intersection(&other.set).copied().collect();
        Subset {
            n_items: set.len(),
            set,
            vector: Vec::new(),
            is_clean: false,
        }
    }

    pub fn contains(&self, i: usize) -> bool {
        self.set.contains(&i)
    }

    fn remove(&mut self, i: usize) -> bool {
        if self.set.remove(&i) {
            self.n_items -= 1;
            self.vector.clear();
            self.is_clean = false;
            true
        } else {
            false
        }
    }

    fn pop(&mut self) -> Option<usize> {
        if !self.is_clean {
            self.clean()
        }
        let i_option = self.vector.pop();
        if let Some(i) = i_option {
            self.set.remove(&i);
            self.n_items -= 1;
        };
        i_option
    }

    fn clean(&mut self) {
        if !self.is_clean {
            for i in &self.set {
                self.vector.push(*i);
            }
            self.is_clean = true;
        }
    }

    /// The number of items in the subset.
    pub fn n_items(&self) -> usize {
        self.n_items
    }

    /// Is the subset empty?
    pub fn is_empty(&self) -> bool {
        self.n_items == 0
    }

    /// A reference to the elements of the set.
    pub fn items(&self) -> &Vec<usize> {
        if !self.is_clean {
            panic!("Subset is not clean.  Please clean it first.");
        }
        &self.vector
    }

    /// A copy of the elements of the set.
    pub fn items_via_copying(&self) -> Vec<usize> {
        if self.is_clean {
            self.vector.clone()
        } else {
            let mut vector = Vec::new();
            for i in &self.set {
                vector.push(*i);
            }
            vector
        }
    }
}

impl fmt::Display for Subset {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut clone = self.clone();
        clone.clean();
        clone.vector.sort_unstable();
        fmt.write_str("{")?;
        let mut str = "";
        for i in clone.vector {
            fmt.write_str(str)?;
            fmt.write_str(&format!("{}", i))?;
            str = ",";
        }
        fmt.write_str("}")?;
        Ok(())
    }
}

impl Default for Subset {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for Subset {
    fn eq(&self, other: &Self) -> bool {
        self.set == other.set
    }
}

#[cfg(test)]
mod tests_subset {
    use super::*;

    #[test]
    fn test_add() {
        let mut s = Subset::new();
        assert!(s.add(0));
        assert!(s.add(1));
        assert!(!s.add(0));
        assert!(!s.add(1));
        assert!(!s.add(0));
        assert_eq!(s.n_items(), 2);
        assert_eq!(s.to_string(), "{0,1}");
    }

    #[test]
    fn test_merge() {
        let mut s1 = Subset::from([2, 1, 6, 2].iter());
        let s2 = Subset::from([0, 2, 1].iter());
        s1.merge(&s2);
        assert_eq!(s1.to_string(), "{0,1,2,6}");
        let mut s3 = Subset::from([7, 2].iter());
        s3.merge(&s1);
        assert_eq!(s3.to_string(), "{0,1,2,6,7}");
    }

    #[test]
    fn test_remove() {
        let mut s = Subset::from([2, 1, 6, 2].iter());
        assert!(!s.remove(0));
        assert!(s.remove(2));
        assert_eq!(s.to_string(), "{1,6}");
    }

    #[test]
    fn test_equality() {
        let s1 = Subset::from([2, 1, 6, 2].iter());
        let s2 = Subset::from([6, 1, 2].iter());
        assert_eq!(s1, s2);
        let s3 = Subset::from([6, 1, 2, 3].iter());
        assert_ne!(s1, s3);
    }
}

/// A data structure holding partitions.
///
pub struct PartitionsHolder {
    data: Vec<i32>,
    n_partitions: usize,
    n_items: usize,
    by_row: bool,
}

impl PartitionsHolder {
    pub fn new(n_items: usize) -> PartitionsHolder {
        PartitionsHolder {
            data: Vec::new(),
            n_partitions: 0,
            n_items,
            by_row: false,
        }
    }

    pub fn with_capacity(capacity: usize, n_items: usize) -> PartitionsHolder {
        PartitionsHolder {
            data: Vec::with_capacity(capacity * n_items),
            n_partitions: 0,
            n_items,
            by_row: false,
        }
    }

    pub fn allocated(n_partitions: usize, n_items: usize, by_row: bool) -> PartitionsHolder {
        PartitionsHolder {
            data: vec![0; n_partitions * n_items],
            n_partitions,
            n_items,
            by_row,
        }
    }

    pub fn enumerated(n_items: usize) -> PartitionsHolder {
        let mut ph = PartitionsHolder::new(n_items);
        for partition in Partition::iter(n_items) {
            ph.push_slice(&partition[..]);
        }
        ph
    }

    pub fn n_partitions(&self) -> usize {
        self.n_partitions
    }

    pub fn n_items(&self) -> usize {
        self.n_items
    }

    pub fn by_row(&self) -> bool {
        self.by_row
    }

    pub fn push_slice(&mut self, partition: &[usize]) {
        assert!(!self.by_row, "Pushing requires that by_row = false.");
        assert_eq!(
            partition.len(),
            self.n_items,
            "Inconsistent number of items."
        );
        for j in partition {
            self.data.push(i32::try_from(*j).unwrap())
        }
        self.n_partitions += 1
    }

    pub fn push_partition(&mut self, partition: &Partition) {
        assert!(!self.by_row, "Pushing requires that by_row = false.");
        assert_eq!(partition.n_items, self.n_items);
        for j in partition.labels() {
            self.data.push(i32::try_from(j.unwrap()).unwrap())
        }
        self.n_partitions += 1
    }

    pub fn view(&mut self) -> PartitionsHolderBorrower {
        PartitionsHolderBorrower::from_slice(
            &mut self.data[..],
            self.n_partitions,
            self.n_items,
            self.by_row,
        )
    }
}

#[doc(hidden)]
pub struct PartitionsHolderBorrower<'a> {
    data: &'a mut [i32],
    n_partitions: usize,
    n_items: usize,
    by_row: bool,
    index: usize,
}

impl std::ops::Index<(usize, usize)> for PartitionsHolderBorrower<'_> {
    type Output = i32;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        if self.by_row {
            &self.data[self.n_partitions * j + i]
        } else {
            &self.data[self.n_items * i + j]
        }
    }
}

impl std::ops::IndexMut<(usize, usize)> for PartitionsHolderBorrower<'_> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        if self.by_row {
            &mut self.data[self.n_partitions * j + i]
        } else {
            &mut self.data[self.n_items * i + j]
        }
    }
}

impl<'a> PartitionsHolderBorrower<'a> {
    pub fn from_slice(
        data: &'a mut [i32],
        n_partitions: usize,
        n_items: usize,
        by_row: bool,
    ) -> Self {
        assert_eq!(data.len(), n_partitions * n_items);
        Self {
            data,
            n_partitions,
            n_items,
            by_row,
            index: 0,
        }
    }

    /// # Safety
    ///
    /// Added for FFI.
    pub unsafe fn from_ptr(
        data: *mut i32,
        n_partitions: usize,
        n_items: usize,
        by_row: bool,
    ) -> Self {
        let data = slice::from_raw_parts_mut(data, n_partitions * n_items);
        Self {
            data,
            n_partitions,
            n_items,
            by_row,
            index: 0,
        }
    }
    pub fn n_partitions(&self) -> usize {
        self.n_partitions
    }

    pub fn n_items(&self) -> usize {
        self.n_items
    }

    pub fn by_row(&self) -> bool {
        self.by_row
    }

    pub fn data(&self) -> &[i32] {
        self.data
    }

    /// # Safety
    ///
    /// You're on your own with the indices.
    pub unsafe fn get_unchecked(&self, (i, j): (usize, usize)) -> &i32 {
        if self.by_row {
            self.data.get_unchecked(self.n_partitions * j + i)
        } else {
            self.data.get_unchecked(self.n_items * i + j)
        }
    }

    /// # Safety
    ///
    /// You're on your own with the indices.
    pub unsafe fn get_unchecked_mut(&mut self, (i, j): (usize, usize)) -> &mut i32 {
        if self.by_row {
            self.data.get_unchecked_mut(self.n_partitions * j + i)
        } else {
            self.data.get_unchecked_mut(self.n_items * i + j)
        }
    }

    pub fn get(&self, i: usize) -> Partition {
        if self.by_row {
            let mut labels = Vec::with_capacity(self.n_items);
            for j in 0..self.n_items {
                labels.push(self.data[self.n_partitions * j + i]);
            }
            Partition::from(&labels[..])
        } else {
            Partition::from(&self.data[(i * self.n_items)..((i + 1) * self.n_items)])
        }
    }

    pub fn get_all(&self) -> Vec<Partition> {
        let mut x = Vec::with_capacity(self.n_partitions);
        for k in 0..self.n_partitions {
            x.push(self.get(k))
        }
        x
    }

    pub fn push_slice(&mut self, partition: &[usize]) {
        assert_eq!(
            partition.len(),
            self.n_items,
            "Inconsistent number of items."
        );
        for (j, v) in partition.iter().enumerate() {
            let v = i32::try_from(*v).unwrap();
            let o = if self.by_row {
                self.n_partitions * j + self.index
            } else {
                self.n_items * self.index + j
            };
            unsafe {
                *self.data.get_unchecked_mut(o) = v;
            }
        }
        self.index += 1;
    }

    pub fn push_partition(&mut self, partition: &Partition) {
        assert!(
            self.index < self.n_partitions,
            "The holder has capacity {} so cannot push with index {}.",
            self.n_partitions,
            self.index
        );
        assert_eq!(
            partition.n_items(),
            self.n_items,
            "Inconsistent number of items."
        );
        for (j, v) in partition.labels().iter().enumerate() {
            let v = i32::try_from(v.unwrap()).unwrap();
            let o = if self.by_row {
                self.n_partitions * j + self.index
            } else {
                self.n_items * self.index + j
            };
            unsafe {
                *self.data.get_unchecked_mut(o) = v;
            }
        }
        self.index += 1;
    }
}

impl fmt::Display for PartitionsHolderBorrower<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.n_partitions {
            let x = self.get(i).to_string();
            fmt.write_str(&x[..])?;
            fmt.write_str("\n")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests_partitions_holder {
    use super::*;

    #[test]
    fn test_partition_holder() {
        let mut phf = PartitionsHolder::new(4);
        assert!(!phf.by_row());
        assert_eq!(phf.n_partitions(), 0);
        phf.push_partition(&Partition::from("AABC".as_bytes()));
        phf.push_partition(&Partition::from("ABCD".as_bytes()));
        phf.push_partition(&Partition::from("AABA".as_bytes()));
        assert_eq!(phf.n_partitions(), 3);
        let phfv = phf.view();
        assert_eq!(phfv.data(), &[0, 0, 1, 2, 0, 1, 2, 3, 0, 0, 1, 0]);

        let mut phf2 = PartitionsHolder::allocated(3, 4, false);
        let mut phfv2 = phf2.view();
        phfv2.push_partition(&Partition::from("AABC".as_bytes()));
        phfv2.push_partition(&Partition::from("ABCD".as_bytes()));
        phfv2.push_partition(&Partition::from("AABA".as_bytes()));
        assert_eq!(phfv2.n_partitions(), 3);
        assert_eq!(phfv.data, phfv2.data);
        unsafe {
            *phfv2.get_unchecked_mut((0, 1)) = 9;
        }
        phfv2[(1, 3)] = 9;
        assert_eq!(phfv2.data(), &[0, 9, 1, 2, 0, 1, 2, 9, 0, 0, 1, 0]);
        assert_eq!(phfv2.get(0), Partition::from("ABCD".as_bytes()));
        assert_eq!(phfv2.get(2), Partition::from("AABA".as_bytes()));

        let mut pht = PartitionsHolder::allocated(3, 4, true);
        assert!(pht.by_row());
        assert_eq!(pht.n_partitions(), 3);
        let mut phtv = pht.view();
        assert_eq!(phtv.data(), &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        phtv.push_partition(&Partition::from("AABC".as_bytes()));
        phtv.push_partition(&Partition::from("ABCD".as_bytes()));
        phtv.push_partition(&Partition::from("AABA".as_bytes()));
        assert_eq!(phtv.n_partitions(), 3);
        assert_eq!(phtv.data(), &[0, 0, 0, 0, 1, 0, 1, 2, 1, 2, 3, 0]);
        unsafe {
            *phtv.get_unchecked_mut((0, 1)) = 9;
        }
        phtv[(1, 3)] = 9;
        assert_eq!(phtv.data(), &[0, 0, 0, 9, 1, 0, 1, 2, 1, 2, 9, 0]);
        assert_eq!(phtv.get(0), Partition::from("ABCD".as_bytes()));
        assert_eq!(phtv.get(2), Partition::from("AABA".as_bytes()));
    }
}

#[doc(hidden)]
#[no_mangle]
pub unsafe extern "C" fn dahl_partition__enumerated(
    n_partitions: i32,
    n_items: i32,
    partitions_ptr: *mut i32,
) {
    let n_partitions = usize::try_from(n_partitions).unwrap();
    let n_items = usize::try_from(n_items).unwrap();
    let mut phv = PartitionsHolderBorrower::from_ptr(partitions_ptr, n_partitions, n_items, true);
    for partition in Partition::iter(n_items) {
        phv.push_slice(&partition[..]);
    }
}

/// A data structure representation a permutation of integers.
///
#[derive(Debug)]
pub struct Permutation(Vec<usize>);

impl Permutation {
    pub fn from_slice(x: &[usize]) -> Option<Self> {
        let mut y = Vec::from(x);
        y.sort_unstable();
        for (i, j) in y.into_iter().enumerate() {
            if i != j {
                return None;
            }
        }
        Some(Self(Vec::from(x)))
    }

    pub fn from_vector(x: Vec<usize>) -> Option<Self> {
        let mut y = x.clone();
        y.sort_unstable();
        for (i, j) in y.into_iter().enumerate() {
            if i != j {
                return None;
            }
        }
        Some(Self(x))
    }

    pub fn natural(n_items: usize) -> Self {
        Self((0..n_items).collect())
    }

    pub fn random<T: Rng>(n_items: usize, rng: &mut T) -> Self {
        let mut perm = Self::natural(n_items);
        perm.shuffle(rng);
        perm
    }

    pub fn shuffle<T: Rng>(&mut self, rng: &mut T) {
        self.0.shuffle(rng)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn slice_until(&self, end: usize) -> &[usize] {
        &self.0[..end]
    }
}

impl std::ops::Index<usize> for Permutation {
    type Output = usize;
    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}

/// A data structure representing a square matrix.
///
pub struct SquareMatrix {
    data: Vec<f64>,
    n_items: usize,
}

impl SquareMatrix {
    pub fn zeros(n_items: usize) -> Self {
        Self {
            data: vec![0.0; n_items * n_items],
            n_items,
        }
    }

    pub fn ones(n_items: usize) -> Self {
        Self {
            data: vec![1.0; n_items * n_items],
            n_items,
        }
    }

    pub fn identity(n_items: usize) -> Self {
        let ni1 = n_items + 1;
        let n2 = n_items * n_items;
        let mut data = vec![0.0; n2];
        let mut i = 0;
        while i < n2 {
            data[i] = 1.0;
            i += ni1
        }
        Self { data, n_items }
    }

    pub fn data(&self) -> &[f64] {
        &self.data[..]
    }

    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data[..]
    }

    pub fn view(&mut self) -> SquareMatrixBorrower {
        SquareMatrixBorrower::from_slice(&mut self.data[..], self.n_items)
    }

    pub fn n_items(&self) -> usize {
        self.n_items
    }
}

pub struct SquareMatrixBorrower<'a> {
    data: &'a mut [f64],
    n_items: usize,
}

impl std::ops::Index<(usize, usize)> for SquareMatrixBorrower<'_> {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.data[self.n_items * j + i]
    }
}

impl std::ops::IndexMut<(usize, usize)> for SquareMatrixBorrower<'_> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.data[self.n_items * j + i]
    }
}

impl<'a> SquareMatrixBorrower<'a> {
    pub fn from_slice(data: &'a mut [f64], n_items: usize) -> Self {
        assert_eq!(data.len(), n_items * n_items);
        Self { data, n_items }
    }

    /// # Safety
    ///
    /// Added for FFI.
    pub unsafe fn from_ptr(data: *mut f64, n_items: usize) -> Self {
        let data = slice::from_raw_parts_mut(data, n_items * n_items);
        Self { data, n_items }
    }

    pub fn n_items(&self) -> usize {
        self.n_items
    }

    /// # Safety
    ///
    /// You're on your own with the indices.
    pub unsafe fn get_unchecked(&self, (i, j): (usize, usize)) -> &f64 {
        self.data.get_unchecked(self.n_items * j + i)
    }

    /// # Safety
    ///
    /// You're on your own with the indices.
    pub unsafe fn get_unchecked_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        self.data.get_unchecked_mut(self.n_items * j + i)
    }

    pub fn data(&self) -> &[f64] {
        self.data
    }

    pub fn data_mut(&mut self) -> &mut [f64] {
        self.data
    }

    pub fn sum_of_triangle(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.n_items {
            for j in 0..i {
                sum += unsafe { *self.get_unchecked((i, j)) };
            }
        }
        sum
    }

    pub fn sum_of_row_subset(&self, row: usize, columns: &[usize]) -> f64 {
        let mut sum = 0.0;
        for j in columns {
            sum += unsafe { *self.get_unchecked((row, *j)) };
        }
        sum
    }
}
