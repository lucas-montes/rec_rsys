use rec_rsys::algorithms::knn::KNN;
use rec_rsys::models::Item;
use rec_rsys::similarity::SimilarityAlgos;

fn setup() -> Vec<Item> {
    vec![
        Item::new(1, vec![0.9193, 0.9097, 0.4990, 0.3292, 0.8811], Some(0.91)),
        Item::new(2, vec![0.9826, 0.9977, 0.6924, 0.7509, 0.7644], Some(0.98)),
        Item::new(3, vec![0.4817, 0.7548, 0.1974, 0.2229, 0.1256], Some(0.48)),
        Item::new(4, vec![0.9376, 0.4734, 0.2254, 0.9728, 0.8401], Some(0.93)),
        Item::new(5, vec![0.7429, 0.3250, 0.5680, 0.2614, 0.4483], Some(0.74)),
        Item::new(6, vec![0.0686, 0.9531, 0.3464, 0.6426, 0.1746], Some(0.06)),
        Item::new(7, vec![0.2442, 0.3728, 0.3096, 0.1398, 0.8162], Some(0.24)),
        Item::new(8, vec![0.3682, 0.9574, 0.0486, 0.8852, 0.1986], Some(0.36)),
        Item::new(9, vec![0.3455, 0.2594, 0.7464, 0.0489, 0.4088], Some(0.34)),
        Item::new(10, vec![0.7193, 0.4097, 0.6990, 0.3292, 0.8811], Some(0.71)),
    ]
}

#[test]
fn test_cosine_knn() {
    let refs: Vec<Item> = setup();
    let new_item = &refs[0];
    let result = KNN::new(new_item.clone(), refs.clone())
        .set_num_neighbors(3)
        .result();
    assert_eq!(result, vec![new_item, &refs[1], &refs[9]]);
    assert_eq!(result[0].result, 1.0000001);
    assert_eq!(result[1].result, 0.969_654_7);
    assert_eq!(result[2].result, 0.94337976);
}

#[test]
fn test_euclidean_knn() {
    let refs: Vec<Item> = setup();
    let new_item = &refs[0];
    let result = KNN::new(new_item.clone(), refs.clone())
        .set_num_neighbors(3)
        .set_algorithm(SimilarityAlgos::Euclidean)
        .result();
    assert_eq!(result, vec![new_item, &refs[1], &refs[9]]);
    assert_eq!(result[0].result, 0.0);
    assert_eq!(result[1].result, 0.4905142);
    assert_eq!(result[2].result, 0.5744563);
}
