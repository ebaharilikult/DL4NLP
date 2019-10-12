package classifier;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * A multi-class perceptron which operates on samples.
 *
 */
public interface PerceptronInterface {

    /**
     * Predict a label for each sample in a list and return the list of predicted
     * labels.
     *
     * @param samples
     * @return
     */
    List<String> predict(List<Sample> samples);

    /**
     * Initialise perceptron by creating weight vectors for each label with all
     * known features.
     *
     * @param features given features
     */
    void initialise(LinkedHashMap<Sample, FeatureVectorInterface> features) throws IOException;

    int getAmountOfClasses();

    /**
     * Train perceptron on feature vectors.
     * 
     * @param evaluationSamples samples to evaluate
     * @param evaluateEachEpoch if the evaluation file should be predicted after
     *                          each epoch
     */
    void train(List<Sample> evaluationSamples, Boolean evaluateEachEpoch);

}
