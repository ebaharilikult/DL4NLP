package classifier;

import main.Config;

import java.util.LinkedHashMap;

public class WeightVector extends SparseVector implements WeightVectorInterface {

    public WeightVector() {
        this.vector = new LinkedHashMap<>(512);
    }

    @Override
    public void update(FeatureVectorInterface featureVectorInterface, Sign sign) {
        this.vector = this.addVector(featureVectorInterface.multiplyScalar(sign.getValue()
                * Config.LEARNING_RATE)).vector;
    }

    @Override
    public LinkedHashMap<Integer, Double> getFeatureWeights() {
        return this.vector;
    }
}
