//! Place to store all the models used to calculate
use ndarray::{iter::Lanes, Array1, Array2, ArrayView1, Axis, Dim};
use num_traits::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::HashMap, iter::Sum};

pub trait Numeric: Float + FromPrimitive + Sum + 'static + std::fmt::Debug {}
impl<T> Numeric for T where T: Float + FromPrimitive + Sum + 'static + std::fmt::Debug {}

#[derive(Debug, Clone, PartialEq)]
pub struct DatasetBase<F> {
    records: Array2<F>,
    weights: Option<Array1<F>>,
    feature_names: Vec<String>,
}

impl<F: Numeric> DatasetBase<F> {
    pub fn new(records: Array2<F>) -> Self {
        Self {
            records,
            weights: None,
            feature_names: Vec::new(),
        }
    }

    pub fn get_row(&self, index: usize) -> ArrayView1<F> {
        self.records.index_axis(Axis(0), index)
        //self.dataset.records.slice(s![item, ..]);
    }

    pub fn rows(
        &self,
    ) -> std::iter::Enumerate<ndarray::iter::AxisIter<F, ndarray::Dim<[usize; 1]>>> {
        self.records.outer_iter().enumerate()
    }
}

#[derive(Debug)]
pub struct ItemResult<F: Numeric>(F, usize);

impl<F: Numeric> ItemResult<F> {
    pub fn new(result: F, index: usize) -> Self {
        Self(result, index)
    }

    pub fn value(&self) -> F {
        self.0
    }
    pub fn index(&self) -> usize {
        self.1
    }
}

impl<F: Numeric> Eq for ItemResult<F> {}

impl<F: Numeric> PartialEq for ItemResult<F> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<F: Numeric> PartialOrd for ItemResult<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.0.partial_cmp(&self.0)
    }
}

impl<F: Numeric> Ord for ItemResult<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}
