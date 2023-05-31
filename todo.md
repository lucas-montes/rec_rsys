1.- Primordial:
- [ ] Fix possible errors in formulas
- [ ] Add tests for each formula to be sure that it's correct
- [ ] Normalize documentation so is the same everywhere
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

How to:
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
