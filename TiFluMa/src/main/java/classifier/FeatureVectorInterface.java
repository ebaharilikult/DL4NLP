package classifier;

import java.util.LinkedHashMap;

/**
 * Vector with featureID-value pairs (and a gold label if known).
 */
public interface FeatureVectorInterface extends SparseVectorInterface {
    /**
     * Returns the vector as a LinkedHashMap. key : id of feature value : feature
     * value
     *
     * @return LinkedHashMap FeatureVector.
     */
    LinkedHashMap<Integer, Double> getFeatureValues();

    String getGoldLabel();

    void setGoldLabel(String goldLabel);
}
