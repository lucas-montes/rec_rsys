# RecSys
A recommender system toolkit with more maths functions. Currently it's only used to learn and improve about this field, but feel free to participate.

[![crates.io](https://img.shields.io/crates/v/rec_rsys.svg)](https://crates.io/crates/rec_rsys)
[![Released API docs](https://docs.rs/rec_rsys/badge.svg)](https://docs.rs/rec_rsys)
![example workflow](https://github.com/lucas-montes/rec_rsys/actions/workflows/ci.yml/badge.svg)

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
