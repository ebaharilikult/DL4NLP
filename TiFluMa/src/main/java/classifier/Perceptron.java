package classifier;

import evaluationMethods.Metrics;
import main.Config;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.Collectors;

public class Perceptron implements PerceptronInterface {

    Logger logger = LoggerFactory.getLogger(Perceptron.class);

    FeatureExtractorInterface featureExtractor;
    LinkedHashMap<String, WeightVector> labelWeightVectorMapping;
    LinkedHashMap<Sample, FeatureVectorInterface> sampleFeatureVectorMapping;

    public Perceptron(FeatureExtractorInterface featureExtractor) {
        this.featureExtractor = featureExtractor;
    }

    public void setFeatures(LinkedHashMap<Sample, FeatureVectorInterface> sampleFeatureVectorMapping) {
        this.sampleFeatureVectorMapping = sampleFeatureVectorMapping;
    }

    @Override
    public void initialise(LinkedHashMap<Sample, FeatureVectorInterface> features) throws IOException {
        logger.debug("initialise >");
        this.sampleFeatureVectorMapping = features;
        this.labelWeightVectorMapping = new LinkedHashMap<String, WeightVector>(1024);

        this.sampleFeatureVectorMapping.forEach((k, v) -> {
            this.labelWeightVectorMapping.putIfAbsent(k.getLabel(), new WeightVector());
        });
    }

    @Override
    public int getAmountOfClasses() {
        if (this.labelWeightVectorMapping == null) {
            return 0;
        }
        return this.labelWeightVectorMapping.keySet().size();
    }

    @Override
    public void train(List<Sample> evaluationSamples, Boolean evaluateEachEpoch) {
        logger.debug("train >");
        Config.LEARNING_RATE_ADAPTATION = 1 / Config.EPOCHS; // currently linear decay
        // double origLearningRate = Config.LEARNING_RATE; <-- alternative implementation for learning rate
        for (int i = 0; i < Config.EPOCHS; i++) {
            // Config.LEARNING_RATE = origLearningRate; <-- alternative implementation for learning rate
            logger.info("Current epoch: {}", i);
            ArrayList<Sample> shuffledSamples = new ArrayList<Sample>(sampleFeatureVectorMapping.keySet());
            ArrayList<String> updateBatchPart1 = new ArrayList<>();
            ArrayList<Sample> updateBatchPart2 = new ArrayList<>();
            Collections.shuffle(shuffledSamples);
            for (Sample sample : shuffledSamples) {
                String prediction = this.predict(sampleFeatureVectorMapping.get(sample));
                // logger.debug("Predicted: " + prediction);
                // logger.debug("Correct: " + sample.getLabel());
                if (!prediction.equals(sample.getLabel())) {
                    // Config.LEARNING_RATE *= Config.LEARNING_RATE_ADAPTATION; <-- alternative implementation for learning rate
                    updateBatchPart1.add(prediction);
                    updateBatchPart2.add(sample);
                }
            }
            for (int j = 0; j < updateBatchPart1.size(); j++) {
                // lower score of wrong answer
                labelWeightVectorMapping.get(updateBatchPart1.get(j)).update(sampleFeatureVectorMapping.get(updateBatchPart2.get(j)), Sign.NEG);
                // raise score of right answer
                labelWeightVectorMapping.get(updateBatchPart2.get(j).getLabel()).update(sampleFeatureVectorMapping.get(updateBatchPart2.get(j)), Sign.POS);
            }
            Config.LEARNING_RATE -= Config.LEARNING_RATE_ADAPTATION;

            if (evaluateEachEpoch) {
                evaluateSamples(evaluationSamples);
            }
        }
        logger.debug("train <");
    }

    /**
     * Evaluate the samples given for intermediate results
     *
     * @param evaluationSamples Evaluation data
     */
    private void evaluateSamples(List<Sample> evaluationSamples) {
        ArrayList<String> evaluationResults = (ArrayList<String>) predict(evaluationSamples);

        Metrics mt = new Metrics();
        mt.setClassifierResults(evaluationResults);
        mt.setGoldLines(evaluationSamples.stream().map(Sample::getLabel).collect(Collectors.toList()));
        try {
            mt.evaluateSamples();
        } catch (IOException e) {
            e.printStackTrace();
        }

        logger.info("Precision:              " + mt.retrieveMacroPrecision());
        logger.info("Recall:                 " + mt.retrieveMacroRecall());
        logger.info("Macro Averaged F-Score: " + mt.retrieveMacroFScore());
        logger.info("Micro Averaged F-Score: " + mt.retrieveMicroFScore());
    }

    @Override
    public List<String> predict(List<Sample> samples) {
        //logger.debug("predict > samples='{}'", samples);

        List<String> results = new ArrayList<String>();
        for (int i = 0; i < samples.size(); i++) {
            //logger.debug("Predicted: {}%", i * 100 / samples.size());
            results.add(predict(samples.get(i)));
        }

        //logger.debug("predict < results='{}'", results);
        return results;
    }

    /**
     * Predicts the label for the given sample
     *
     * @param sample sample to predict
     * @return predicted label
     */
    private String predict(Sample sample) {

        FeatureVectorInterface sampleFeatureVector = featureExtractor.createFeatureVector(sample);
        return predict(sampleFeatureVector);
    }

    /**
     * Predicts the label that fits the most to the given feature vector
     *
     * @param featureVector feature vector to get the label for
     * @return predicted label
     */
    private String predict(FeatureVectorInterface featureVector) {
        String predictedLabel = "";
        double maxScore = Double.NEGATIVE_INFINITY;

        for (String label : labelWeightVectorMapping.keySet()) {
            double predictedScore = labelWeightVectorMapping.get(label).scalarProduct((SparseVector) featureVector);

            if (predictedScore > maxScore) {
                maxScore = predictedScore;
                predictedLabel = label;
            }
        }
        return predictedLabel;
    }

}
