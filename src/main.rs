#![allow(dead_code, unused)]
#![deny(
    missing_docs,
    nonstandard_style,
    unused_variables,
    unused_mut,
    unused_parens,
    rust_2018_idioms,
    rust_2018_compatibility,
    future_incompatible,
    missing_copy_implementations
)]
//! Awesome
mod accuracy;
mod algorithms;
mod matrix;
mod models;
mod similarity;
mod statistics;
mod testing_tools;
mod utils;

use testing_tools::{compare_execution_times, create_vector};

use std::cell::RefCell;
use std::rc::Rc;
fn main() {
    // for v in [5, 6, 7, 8, 9, 10, 20] {
    //     let mut matrix = Vec::new();
    //     for _ in 0..v {
    //         matrix.push(create_vector(v, -1000.0, 1000.0));
    //     }

    //     let m1: Vec<Vec<f64>> = matrix.clone();
    //     let m2: Vec<Vec<f64>> = matrix.clone();
    //     let _m3: Vec<Vec<f64>> = matrix.clone();
    //     let _m4: Vec<Vec<f64>> = matrix.clone();
    //     let _m5: Vec<Vec<f64>> = matrix.clone();

    //     let functions: Vec<(&str, Rc<RefCell<dyn Fn()>>)> = vec![
    //         (
    //             "covariance",
    //             Rc::new(RefCell::new(move || {
    //                 covariance(&m1);
    //             })),
    //         ),
    //         (
    //             "covariance_matrix",
    //             Rc::new(RefCell::new(move || {
    //                 covariance_matrix(&m2);
    //             })),
    //         ),
    //     ];
    //     println!("------Execution for {:?}-------", v);
    //     compare_execution_times(100, functions);
    //     println!("---------------------------------");
    //     println!("---------------------------------");
    //     println!("---------------------------------");
    // }
}
