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
- [ ] Create two types of docs. One in separated .md file with extense explanation and math examples. And the second one more for "code use"
- [ ] Fix typos
- [ ] Create a trait for the similarities
- [ ] Share this trait with the struct representing Items and Users

2.- Nice to have:
- [ ] Add more docs in .md related
- [ ] Add tests in the docs
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
/// ## Explanation:
/// [Explanation of the mathematical concept]
/// 
/// ## Formula:
/// $$ [Mathematical formula in raw katex format] $$
///
/// ### Where:
/// * [Definition of each component of the formula]
```
- *Order :*
<br>Keep the related concepts together
