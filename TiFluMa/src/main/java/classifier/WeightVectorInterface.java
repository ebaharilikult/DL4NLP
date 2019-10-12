package classifier;

import java.util.LinkedHashMap;

/**
 * Weight vector with featureID-value pairs.
 */
public interface WeightVectorInterface extends SparseVectorInterface {
    /**
     * Updates the weight vector according to the feature vector and the sign of
     * the update using the learning rate.
     *
     * @param featureVectorInterface
     *            The feature vector.
     * @param sign
     *            The update direction.
     */
    void update(FeatureVectorInterface featureVectorInterface, Sign sign);

    /**
     * Returns the vector as a LinkedHashMap. key : id of feature value : feature
     * value
     *
     * @return LinkedHashMap WeightVector.
     */
    LinkedHashMap<Integer, Double> getFeatureWeights();
}
