## Formula:
$$ \text{percentile} = L + (U - L) \cdot (P - \text{floor}(P)) $$
## Explanation:
The percentile calculation is based on linear interpolation. Given a sorted
set of samples, the percentile formula calculates the value at the desired percentile (P)
by interpolating between the lower value (L) and the upper value (U) closest to the rank
corresponding to the desired percentile. The interpolation factor is determined by the
fractional part of the rank.