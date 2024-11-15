//! Place to store all the models used to calculate
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
/// Generic model to save the results
// Similarity struct: used to store the result of the similarities calculation
// struct Result {
//      id: u32,
//      result: f32,
//      confidence: f32,
//      explanation: String,
//      date: String / DateTime,
//      algorithm: String,
// }
use std::collections::HashMap;

/// Generic model to perform calculations
#[derive(Debug, Clone, Serialize)]
pub struct Item {
    /// Identifier
    pub id: u32,
    /// Vector of values used to calculate
    pub values: Vec<f32>,
    /// The result of the similarity calculated
    /// This could be changed into a new struct with more info
    pub result: f32,
    // pub result: Result,
}

impl Item {
    fn default() -> Self {
        Item {
            id: 0,
            values: vec![0.0],
            result: f32::NAN,
        }
    }

    /// New
    pub fn new(id: u32, values: Vec<f32>, result: Option<f32>) -> Self {
        Item {
            id,
            values,
            result: result.unwrap_or(f32::NAN),
        }
    }

    pub fn result(mut self, r: f32) -> Self {
        self.result = r;
        self
    }
}

impl PartialEq for Item {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<'a> PartialEq<Item> for &'a Item {
    fn eq(&self, other: &Item) -> bool {
        *self == other
    }
}

impl<'a> PartialEq<&'a Item> for Item {
    fn eq(&self, other: &&'a Item) -> bool {
        self.id == other.id
    }
}

pub trait ItemAdapter {
    fn to_item(&self) -> Item;
    fn create_values(&self) -> Vec<f32>;
    fn get_references(&self) -> Vec<Item>;
}

#[async_trait]
pub trait AsyncItemAdapter {
    async fn to_item(&self) -> Item;
    async fn create_values(&self) -> Vec<f32>;
    async fn get_references(&self) -> Vec<Item>;
}

pub fn one_hot_encode(labels: &[&str]) -> HashMap<String, Vec<f32>> {
    let mut encoding_map: HashMap<String, Vec<f32>> = HashMap::new();

    for label in labels {
        let encoding: Vec<f32> = labels
            .iter()
            .map(|&l| if l == *label { 1.0 } else { 0.0 })
            .collect();

        encoding_map.insert(label.to_string(), encoding);
    }

    encoding_map
}

pub fn sum_encoding_vectors(
    encoding_map: &HashMap<String, Vec<f32>>,
    values: &[String],
) -> Vec<f32> {
    let mut sum_vector = vec![0.0; encoding_map.values().next().map_or(0, |v| v.len())];

    for value in values {
        if let Some(encoding) = encoding_map.get(value) {
            for (sum_value, &enc_value) in sum_vector.iter_mut().zip(encoding.iter()) {
                *sum_value += enc_value;
            }
        }
    }

    sum_vector
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_hot_encode() {
        let labels = vec!["red", "blue", "green"];
        let expected: HashMap<String, Vec<f32>> = HashMap::from([
            (String::from("blue"), vec![0.0, 1.0, 0.0]),
            (String::from("green"), vec![0.0, 0.0, 1.0]),
            (String::from("red"), vec![1.0, 0.0, 0.0]),
        ]);
        assert_eq!(one_hot_encode(&labels), expected);
    }

    #[test]
    fn test_sum_encoding_vectors() {
        let labels = vec!["red".to_string(), "blue".to_string()];
        let encoding_map: HashMap<String, Vec<f32>> = HashMap::from([
            (String::from("blue"), vec![0.0, 1.0, 0.0]),
            (String::from("green"), vec![0.0, 0.0, 1.0]),
            (String::from("red"), vec![1.0, 0.0, 0.0]),
        ]);
        assert_eq!(
            sum_encoding_vectors(&encoding_map, &labels),
            vec![1.0, 1.0, 0.0]
        );
    }
}
