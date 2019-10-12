package classifier;

import java.util.LinkedHashMap;
import java.util.HashSet;

/**
 * Vector with featureID-value pairs. a missing feature is mathematically
 * equivalent to the value 0 for that feature.
 */
public class SparseVector implements SparseVectorInterface {

    /**
     * key : id of feature value : value
     */
    LinkedHashMap<Integer, Double> vector = new LinkedHashMap<>();

    public SparseVector() {

    }

    @Override
    public void addFeature(int featureID) {
        if (this.vector.containsKey(featureID)) {
            throw new IllegalArgumentException("Tried to initialise feature that already exists");
        }
        this.setFeatureValue(featureID, 0.0);
    }

    @Override
    public void setFeatureValue(int featureID, double featureValue) {
        this.vector.put(featureID, featureValue);
    }

    @Override
    public Double getFeatureValue(int featureID) {
        return this.vector.get(featureID);
    }

    @Override
    public SparseVector multiplyVector(SparseVector otherSparseVector) {
        SparseVector newSparseVector = new SparseVector();
        /*
         * Intersection to avoid missing keys in one set during multiplication. This
         * enables sparse vectors
         */
        HashSet<Integer> intersection = new HashSet<>(this.vector.keySet());
        intersection.retainAll(otherSparseVector.vector.keySet());
        for (int key : intersection) {
            newSparseVector.setFeatureValue(key, this.vector.get(key) * otherSparseVector.vector.get(key));
        }
        return newSparseVector;
    }

    @Override
    public SparseVector multiplyScalar(double scalar) {
        SparseVector newSparseVector = new SparseVector();
        for (int key : this.vector.keySet()) {
            newSparseVector.setFeatureValue(key, this.vector.get(key) * scalar);
        }
        return newSparseVector;
    }

    @Override
    public SparseVector addVector(SparseVector otherSparseVector) {
        SparseVector newSparseVector = new SparseVector();
        /*
         * union because we deal with sparse vectors
         */
        HashSet<Integer> union = new HashSet<>(this.vector.keySet());
        union.addAll(otherSparseVector.vector.keySet());
        for (int key : union) {
            newSparseVector.setFeatureValue(key,
                    this.vector.getOrDefault(key, 0.0) + otherSparseVector.vector.getOrDefault(key, 0.0));
        }
        return newSparseVector;
    }

    @Override
    public double scalarProduct(SparseVector otherSparseVector) {
        return otherSparseVector.vector.keySet().parallelStream()
                .mapToDouble(a -> otherSparseVector.vector.get(a) * this.vector.getOrDefault(a, 0.0)).sum();
    }

    public String toString() {
        String repr = "";
        for (Integer key : this.vector.keySet()) {
            repr += key + " : " + this.vector.get(key) + "\n";
        }
        return repr;
    }
}
