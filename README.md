# RecSys
A recommender system toolkit with more maths functions. Currently it's only used to learn and improve about this field, but feel free to participate.

[![crates.io](https://img.shields.io/crates/v/rec_rsys.svg)](https://crates.io/crates/rec_rsys)
[![Released API docs](https://docs.rs/rec_rsys/badge.svg)](https://docs.rs/rec_rsys)
![example workflow](https://github.com/lucas-montes/rec_rsys/actions/workflows/ci.yml/badge.svg)

## Example
You can find a working example [here](https://github.com/lucas-montes/rsysaas)

A simple implementation would be:

```rust
// src/s/company.rs

use rec_rsys::models::{one_hot_encode, ItemAdapter, Item};

pub struct Company {
    pub id: u32,
    pub ticker: String,
    pub sector: String,
    pub industry: String,
    pub exchange: String,
    pub country: String,
    pub adj: String,
    pub growth: f32,
}

impl ItemAdapter for Company {
    fn to_item(&self) -> Item {
        Item::new(self.id, self.create_values(), None)
    }
    fn create_values(&self) -> Vec<f32> {
        let mut values = vec![self.growth];
        [
            self.encode_sector(),
            self.encode_industry(),
            self.encode_exchange(),
            self.encode_country(),
            self.encode_adjs(),
        ]
        .iter()
        .for_each(|f| values.extend(f));
        values
    }
    fn get_references(&self) -> Vec<Item> {
        match self.get_references_query() {
            Ok(items) => items.then(|c| c.to_item()).collect::<Vec<Item>>(),
            Err(_e) => vec![],
        }
    }
}

impl Company {
    fn get_references_query(&self) -> Result<Vec<Company>, CRUDError> {
        let query = Orm::new()
            .select("id, sector, industry, exchange, country, adj, growth")
            .from(&Self::table())
            .where_clause()
            .not_equal("id", &self.id.to_string())
            .ready();
        let rows = sqlx::query_as::<_, Self>(&query)
            .fetch_all(&mut Self::connect());
        match rows {
            Ok(json) => Ok(json),
            Err(_e) => Err(CRUDError::WrongParameters),
        }
    }

    fn encode_sector(&self) -> Vec<f32> {
        let sectors = vec![
            "Healthcare",
            "Unknown",
            "Automotive",
            "Technology",
            "Communication Services",
            "Basic Materials",
            "Consumer Cyclical",
            "Industrials",
            "Financial Services",
            "Energy",
            "Utilities",
            "Real Estate",
            "Consumer Defensive",
        ];
        match one_hot_encode(&sectors).get(&self.sector) {
            Some(val) => val.to_vec(),
            None => panic!(),
        }
    }

    // rest of methods ...
}
```

```rust
// src/recommendations/company.rs
use rec_rsys::{algorithms::knn::KNN, models::Item, similarity::SimilarityAlgos};
use super::models::company::Company;

pub struct Recommendation {
    prod_id: u32,
    result: f32,
}

fn generate_recommendations(id: u32, num_recommendations: u8) -> Vec<Recommendation> {
        let company = Company::get(id);
        Self::calculate_recommendations(company.to_item(), company.get_references(), num_recommendations)
    }

fn calculate_recommendations(
        item: Item,
        references: Vec<Item>,
        num_recs: u8,
    ) -> Vec<Recommendation> {
        let knn = KNN::new(item, references, num_recs);
        knn.result(SimilarityAlgos::Cosine)
            .into_iter()
            .map(|item| Recommendation{item.id, item.result})
            .collect()
    }
```

```rust
// src/main.rs
mod models;
mod recommendations;

use recommendations::generate_recommendations;

fn main() {
    let recs = generate_recommendations(1, 5);
}
```

## Improvements
1.- Primordial:
- [ ] Fix possible errors in formulas
- [ ] Add tests for each formula to be sure that it's correct
- [ ] Normalize documentation so is the same everywhere
- [X] Create two types of docs. One in separated .md file with extense explanation and math examples. And the second one more for "code use"
- [ ] Fix typos
- [ ] Add benches for the formulas and overall functions

2.- Nice to have:
- [ ] Add more docs in .md related
- [ ] Add tests in the docs
- [ ] Normalize the results. Either 0 or 1 should represent 100% of similarity depending of the formula
- [ ] Convert the results into structs with more information
- [ ] Improve the code snippets. (The title can be the method's name)
- [ ] Make it async

3.- Final steps:
- [ ] Accept incoming data
- [ ] Convert incoming data into structs?
- [ ] Process data and get rankings
- [ ] Check ranking accuracy
- [ ] Run multiples algorithms at the same time

4.- Future nice to have:
- [ ] Save data and results
- [ ] Create some sort of "cache" to avoid multiples recalculations
- [ ] Use ndarrays of some sort of efficient sci-library
- [ ] Compare the performance and results between Generic types, f32 and f64.

## How to:
- *Docs structure :*
```rust
/// # [Name of the concept]
/// [Small explanation of the function]
///
/// ## Parameters:
/// * `[Parameter of the function]`: [Small explanation]
///
/// ## Returns:
/// * [What does the function returns]
/// 
/// ## Examples:
/// [Examples]
/// 
#[doc = include_str!("../docs/example/example.md")]
pub fn example(){}
```
In the folder docs/ create a new .md file with the mathematical formula, explanation and examples if necessary.
```markdown
# [Name of the concept]

## Explanation:
[Explanation of the mathematical concept]

## Formula:
$$ [Mathematical formula in raw katex format] $$

### Where:
* [Definition of each component of the formula]
```
- *Order :*
<br>Keep the related concepts together
