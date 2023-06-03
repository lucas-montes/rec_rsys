The Pearson Correlation and Pearson Baseline Similarity are both measures used to compute the similarity between two variables or entities. However, they are used in different contexts and with different purposes. Here's a brief explanation of the differences between them:

    Pearson Correlation:
        Purpose: Measures the linear relationship between two variables.
        Calculation: Computes the covariance between the two variables divided by the product of their standard deviations.
        Range: The correlation coefficient ranges from -1 to 1, where -1 indicates a perfect negative correlation, 1 indicates a perfect positive correlation, and 0 indicates no linear correlation.
        Application: Commonly used in statistics and data analysis to determine the strength and direction of the relationship between two variables.

    Pearson Baseline Similarity:
        Purpose: Measures the similarity between two entities (e.g., users, items) based on their ratings in collaborative filtering.
        Calculation: Similar to Pearson Correlation, but it involves adjusting the ratings by subtracting the respective user/item baseline (mean) ratings.
        Range: The similarity score typically ranges from -1 to 1, similar to the correlation coefficient.
        Application: Primarily used in collaborative filtering recommendation systems to identify similar users or items based on their ratings. The baseline adjustment helps to account for user/item biases.

In summary, while Pearson Correlation focuses on measuring the linear relationship between variables, Pearson Baseline Similarity is specifically designed for similarity computation in collaborative filtering scenarios, where rating biases are considered by adjusting the ratings to a baseline level.


## Correlation

$$
\text{correlation} = \frac{\text{cov}(X, Y)}{\text{std}(X) \cdot \text{std}(Y)}
$$

## Similarity

$$
\text{similarity} = \frac{\text{cov}(X_{\text{adj}}, Y_{\text{adj}})}{\text{std}(X_{\text{adj}}) \cdot \text{std}(Y_{\text{adj}})}
$$