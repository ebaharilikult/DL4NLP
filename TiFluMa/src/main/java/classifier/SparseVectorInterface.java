package classifier;

public interface SparseVectorInterface {
    /**
     * Adds a new feature and initialises it with zero.
     *
     * @param featureID ID of the feature to add.
     */
    void addFeature(int featureID);

    /**
     * Sets the value behind the featureID provided.
     *
     * @param featureID    ID of the feature.
     * @param featureValue Value of the feature.
     */
    void setFeatureValue(int featureID, double featureValue);

    /**
     * Get the value behind the featureID provided.
     *
     * @param featureID ID of the feature.
     * @return The value of the feature.
     */
    Double getFeatureValue(int featureID);

    /**
     * Multiplies another vector to the vector.
     *
     * @param otherVector Vector that will be multiplied.
     * @return Resulting vector.
     */
    SparseVector multiplyVector(SparseVector otherVector);

    /**
     * Multiplies a scalar to the vector.
     *
     * @param scalar Scalar to be multiplied.
     * @return Resulting vector.
     */
    SparseVector multiplyScalar(double scalar);

    /**
     * Adds another vector to the vector (mathematically).
     *
     * @param otherVector Vector to be added.
     * @return Resulting vector.
     */
    SparseVector addVector(SparseVector otherVector);

    /**
     * Calculates the scalar product of two vectors.
     *
     * @param otherVector
     * @return
     */
    double scalarProduct(SparseVector otherVector);

}
