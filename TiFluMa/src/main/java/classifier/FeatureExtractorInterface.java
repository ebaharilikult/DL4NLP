package classifier;

import java.util.LinkedHashMap;

/**
 * Extract features from samples.
 */
public interface FeatureExtractorInterface {

        /**
         * Creating a new feature vector, adding all its dimensions and filling each
         * dimension with the feature values extracted from the sample provided.
         *
         * @param sample The sample object to extract the features from.
         * @return The new feature vector that represents the sample parameter
         */
        FeatureVectorInterface createFeatureVector(Sample sample);

        /**
         * Gives a new feature ID that has never been used before. Need to be careful
         * not to call this one too often, after the first featurevector is generated,
         * this method should be never called again to maintain integrity of the
         * dimensions.
         *
         * @return A previously never used ID
         */
        int getNewFeatureId();

        /**
         * Getting the feature ID of a word
         *
         * @param word The word to look up
         * @return Feature ID
         */
        int lookupWordID(String word);

        /**
         * Getter for the Lexicon
         *
         * @return The mapping of all known words and their IDs
         */
        LinkedHashMap<String, Integer> getLexicon();

        /**
         * Add a word to the lexicon with an ID that has never been used before
         *
         * @param word Word to be added into the lexicon
         * @param ID   Unique ID of the word
         */
        void addToLexicon(String word, int ID);

        /**
         * Creates a count vector of words occurring in the given text
         *
         * @param text A text to look at token by token and add them to the feature
         * @return A section for the feature vector
         */
        LinkedHashMap<Integer, Double> createWordCountVector(String text, boolean preprocess);

        /**
         * Makes a new feature vector with only topN features
         *
         * @param sampleFeatureVectorMapping The old feature vectors
         * @param minFreq                    The minimum frequency of occurrence we want
         *                                   to accept
         * @return The reduced feature vectors
         */
        LinkedHashMap<Sample, FeatureVectorInterface> onlyFrequentFeatures(
                        LinkedHashMap<Sample, FeatureVectorInterface> sampleFeatureVectorMapping, int minFreq);

        /**
         * Makes a new feature vector with only topN features
         *
         * @param sampleFeatureVectorMapping The old feature vectors
         * @param topN                       The amount of features we want to keep
         * @return The reduced feature vectors
         */
        LinkedHashMap<Sample, FeatureVectorInterface> topFrequentFeatures(
                        LinkedHashMap<Sample, FeatureVectorInterface> sampleFeatureVectorMapping, int topN);

        /**
         * Remove uninformative features
         *
         * @param sampleFeatureVectorMapping The old feature vectors
         * @param minFreq                    The minimum corpus frequency we want to
         *                                   accept
         * @return The reduced feature vectors
         */
        LinkedHashMap<Sample, FeatureVectorInterface> removeHapaxLegomena(
                        LinkedHashMap<Sample, FeatureVectorInterface> sampleFeatureVectorMapping, int minFreq);

        /**
         * Remove the most uninformative features (corpus frequency < 2)
         *
         * @param sampleFeatureVectorMapping The old feature vectors
         * @return The reduced feature vectors
         */
        LinkedHashMap<Sample, FeatureVectorInterface> removeHapaxLegomena(
                        LinkedHashMap<Sample, FeatureVectorInterface> sampleFeatureVectorMapping);

        /**
         * Returns the last used feature id.
         *
         * @return the last used Id
         */
        int getFeatureIDCounter();
}
