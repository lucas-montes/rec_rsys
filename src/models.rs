//! Generic models to perform calculations
struct User {
    pub items: Vec<Item>,
}

struct Item {
    pub id: i16,
    pub rating: f64,
}
