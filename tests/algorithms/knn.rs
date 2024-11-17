use ndarray::array;
use rec_rsys::{
    algorithms::knn::KNearestNeighbors, models::DatasetBase,
    similarity::SimilarityAlgorithm,
};

fn setup() -> DatasetBase<f32> {
    let records = array![
        [0.9193, 0.9097, 0.4990, 0.3292, 0.8811],
        [0.9826, 0.9977, 0.6924, 0.7509, 0.7644],
        [0.4817, 0.7548, 0.1974, 0.2229, 0.1256],
        [0.9376, 0.4734, 0.2254, 0.9728, 0.8401],
        [0.7429, 0.3250, 0.5680, 0.2614, 0.4483],
        [0.0686, 0.9531, 0.3464, 0.6426, 0.1746],
        [0.2442, 0.3728, 0.3096, 0.1398, 0.8162],
        [0.3682, 0.9574, 0.0486, 0.8852, 0.1986],
        [0.3455, 0.2594, 0.7464, 0.0489, 0.4088],
        [0.7193, 0.4097, 0.6990, 0.3292, 0.8811],
    ];
    DatasetBase::new(records)
}

#[test]
fn test_default_knn() {
    let dataset = setup();
    let model = KNearestNeighbors::new(dataset);
    let result = model.predict(0).results();
    assert_eq!(result.len(), 4);
    assert_eq!(result[0].value(), 1.000_000_1);
    assert_eq!(result[1].value(), 0.969_654_7);
    assert_eq!(result[2].value(), 0.943_379_76);
    assert_eq!(result[3].value(), 0.930_861_53);
    assert_eq!(result[0].index(), 0);
    assert_eq!(result[1].index(), 1);
    assert_eq!(result[2].index(), 9);
    assert_eq!(result[3].index(), 4);

    let result = model.predict(1).results();
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].value(), 1.0);
    assert_eq!(result[1].value(), 0.969_654_7);
    assert_eq!(result[0].index(), 1);
    assert_eq!(result[1].index(), 0);
}

#[test]
fn test_euclidean_knn() {
    let dataset = setup();
    let model = KNearestNeighbors::new(dataset)
        .set_num_neighbors(10)
        .set_early_return_threshold(None)
        .set_algorithm(SimilarityAlgorithm::EuclideanDistance);
    let result = model.predict(0).results();
    assert_eq!(result.len(), 10);
    assert_eq!(result[0].value(), 1.0);
    assert_eq!(result[1].value(), 0.50948584);
    assert_eq!(result[2].value(), 0.42554373);
    assert_eq!(result[0].index(), 0);
    assert_eq!(result[1].index(), 1);
    assert_eq!(result[2].index(), 9);
    assert_eq!(result[3].index(), 4);
}
