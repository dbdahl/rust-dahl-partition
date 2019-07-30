use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;

// ## Partition

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
    pub fn new(n_items: usize) -> Partition {
        Partition {
            n_items,
            n_allocated_items: 0,
            subsets: Vec::new(),
            labels: vec![None; n_items],
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
    pub fn from<T>(labels: &[T]) -> Partition
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
        Partition {
            n_items,
            n_allocated_items: n_items,
            subsets,
            labels: new_labels,
        }
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

    /// A reference to a vector of length `n_items` whose elements are `None` for items that are
    /// not allocated and, for items that are allocated, `Some(subset_index)` where `subset_index`
    /// is the index of the subset to which the item is allocated.
    pub fn labels(&self) -> &Vec<Option<usize>> {
        &self.labels
    }

    /// A reference to a vector of length `n_subsets` giving subsets, some of which may be empty.
    pub fn subsets(&self) -> &Vec<Subset> {
        &self.subsets
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

    /// Add a new subset containg `item_index` to the partition.
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
    pub fn add(&mut self, item_index: usize) -> &mut Partition {
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
    pub fn add_with_index(&mut self, item_index: usize, subset_index: usize) -> &mut Partition {
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
    pub fn remove(&mut self, item_index: usize) -> &mut Partition {
        self.check_item_index(item_index);
        self.remove_engine(item_index, self.check_allocated(item_index));
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
    pub fn remove_with_index(&mut self, item_index: usize, subset_index: usize) -> &mut Partition {
        self.check_item_index(item_index);
        self.check_item_in_subset(item_index, subset_index);
        self.remove_engine(item_index, subset_index);
        self
    }

    /// Transfers item `item_index` to a new empty subset.
    pub fn transfer(&mut self, item_index: usize) -> &mut Partition {
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
    ) -> &mut Partition {
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
    /// assert_eq!(partition.to_string(), "_ 0 _");
    /// partition.add(0);
    /// assert_eq!(partition.to_string(), "1 0 _");
    /// partition.canonicalize();
    /// assert_eq!(partition.to_string(), "0 1 _");
    /// ```
    pub fn canonicalize(&mut self) -> &mut Partition {
        if self.n_allocated_items == 0 {
            return self;
        }
        let mut new_labels = vec![None; self.n_items];
        let mut next_label: usize = 0;
        for i in 0..self.n_items {
            match new_labels[i] {
                Some(_) => (),
                None => match self.labels[i] {
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
            } else {
                if y.is_empty() {
                    Ordering::Less
                } else {
                    new_labels[x.vector[0]]
                        .unwrap()
                        .cmp(&new_labels[y.vector[0]].unwrap())
                }
            }
        });
        while self.subsets.len() > 0 && self.subsets.last().unwrap().is_empty() {
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
            panic!(format!(
                "Attempt to allocate item {} when only {} are available.",
                item_index, self.n_items
            ));
        };
    }

    fn check_subset_index(&self, subset_index: usize) {
        if subset_index >= self.n_subsets() {
            panic!(format!(
                "Attempt to allocate to subset {} when only {} are available.",
                subset_index,
                self.n_subsets()
            ));
        };
    }

    fn check_allocated(&self, item_index: usize) -> usize {
        match self.labels[item_index] {
            Some(subset_index) => subset_index,
            None => panic!(format!("Item {} is not allocated.", item_index)),
        }
    }

    fn check_not_allocated(&self, item_index: usize) {
        match self.labels[item_index] {
            Some(j) => panic!(format!(
                "Item {} is already allocated to subset {}.",
                item_index, j
            )),
            None => (),
        };
    }

    fn check_item_in_subset(&self, item_index: usize, subset_index: usize) {
        match self.labels[item_index] {
            Some(j) => {
                if j != subset_index {
                    panic!(format!(
                        "Item {} is already allocated to subset {}.",
                        item_index, j
                    ));
                };
            }
            None => panic!(format!(
                "Item {} is not allocated to any subset.",
                item_index
            )),
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
mod partition_tests {
    use super::*;

    #[test]
    fn complete() {
        let mut p = Partition::new(6);
        p.add(2);
        for i in &[3, 4, 5] {
            p.add_with_index(*i, 0);
        }
        assert_eq!(p.to_string(), "_ _ 0 0 0 0");
        assert_eq!(p.subsets_are_exhaustive(), false);
        p.add(1);
        p.add_with_index(0, 1);
        p.canonicalize();
        assert_eq!(p.to_string(), "0 0 1 1 1 1");
        p.remove_with_index(2, 1);
        p.canonicalize();
        assert_eq!(p.to_string(), "0 0 _ 1 1 1");
        assert_ne!(p.subsets_are_exhaustive(), true);
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
    fn canonicalize() {
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
    fn from() {
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
    fn equality() {
        let p0 = Partition::from("ABCAADAABD".as_bytes());
        let p1 = Partition::from("ABCAADAABD".as_bytes());
        let p2 = Partition::from("ABCRRRRRRD".as_bytes());
        assert_eq!(p0, p1);
        assert_ne!(p0, p2);
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
    fn new() -> Subset {
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

    fn add(&mut self, i: usize) -> bool {
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
    pub fn items(&mut self) -> &Vec<usize> {
        if !self.is_clean {
            self.clean();
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

impl PartialEq for Subset {
    fn eq(&self, other: &Self) -> bool {
        self.set == other.set
    }
}

#[cfg(test)]
mod subset_tests {
    use super::*;

    #[test]
    fn add() {
        let mut s = Subset::new();
        assert_eq!(s.add(0), true);
        assert_eq!(s.add(1), true);
        assert_eq!(s.add(0), false);
        assert_eq!(s.add(1), false);
        assert_eq!(s.add(0), false);
        assert_eq!(s.n_items(), 2);
        assert_eq!(s.to_string(), "{0,1}");
    }

    #[test]
    fn merge() {
        let mut s1 = Subset::from([2, 1, 6, 2].iter());
        let mut s2 = Subset::from([0, 2, 1].iter());
        s1.merge(&mut s2);
        assert_eq!(s1.to_string(), "{0,1,2,6}");
        let mut s3 = Subset::from([7, 2].iter());
        s3.merge(&mut s1);
        assert_eq!(s3.to_string(), "{0,1,2,6,7}");
    }

    #[test]
    fn remove() {
        let mut s = Subset::from([2, 1, 6, 2].iter());
        assert_eq!(s.remove(0), false);
        assert_eq!(s.remove(2), true);
        assert_eq!(s.to_string(), "{1,6}");
    }

    #[test]
    fn equality() {
        let s1 = Subset::from([2, 1, 6, 2].iter());
        let s2 = Subset::from([6, 1, 2].iter());
        assert_eq!(s1, s2);
        let s3 = Subset::from([6, 1, 2, 3].iter());
        assert_ne!(s1, s3);
    }
}
