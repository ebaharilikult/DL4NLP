package classifier;

import java.util.LinkedHashMap;

public class FeatureVector extends SparseVector implements FeatureVectorInterface {

    private String goldLabel;

    public FeatureVector() {
        this.vector = new LinkedHashMap<>(512);
        this.goldLabel = null;
    }

    @Override
    public LinkedHashMap<Integer, Double> getFeatureValues() {
        return this.vector;
    }

    @Override
    public String getGoldLabel() {
        return this.goldLabel;
    }

    @Override
    public void setGoldLabel(String goldLabel) {
        this.goldLabel = goldLabel;
    }

}
