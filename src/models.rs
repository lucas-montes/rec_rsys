//! Place to store all the models used to calculate

/// Generic model to perform calculations
#[derive(Debug, Clone)]
pub struct Item {
    /// Identifier
    pub id: u32,
    /// Vector of values used to calculate
    pub values: Vec<f32>,
    /// The result of the similarity calculated
    pub result: f32,
}

impl Item {
    /// New
    pub fn new(id: u32, values: Vec<f32>, result: f32) -> Self {
        Item { id, values, result }
    }
    // pub fn new(id: u32, values: Vec<f32>) -> Self {
    //     Item {
    //         id,
    //         values,
    //         result: f32::NAN,
    //     }
    // }
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
        self == *other
    }
}
