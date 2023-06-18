//! Place to store all the models used to calculate

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
#[derive(Debug, Clone)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_hot_encode() {
        let labels = vec!["red", "blue", "green"];
        let excpected: HashMap<String, Vec<f32>> = HashMap::from([
            (String::from("blue"), vec![0.0, 1.0, 0.0]),
            (String::from("green"), vec![0.0, 0.0, 1.0]),
            (String::from("red"), vec![1.0, 0.0, 0.0]),
        ]);
        assert_eq!(one_hot_encode(&labels), excpected);
    }
}
