package classifier;

import main.Config;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class FeatureExtractor implements FeatureExtractorInterface {

    private LinkedHashMap<String, Integer> lexicon;
    private LinkedHashMap<String, SparseVector> wordVectors;
    private int featureIDCounter;
    private opennlp.tools.postag.POSModel pos_model;
    private opennlp.tools.postag.POSTaggerME pos_tagger;
    private ClassLoader loader;

    /**
     * Constructor of the feature extractor initialises some features into the
     * lexicon with feature IDs
     *
     * @throws IOException
     */
    public FeatureExtractor() throws IOException {
        this.loader = ClassLoader.getSystemClassLoader();
        this.lexicon = new LinkedHashMap<>(2048);
        this.featureIDCounter = 0;
        this.lexicon.put("BIAS", this.featureIDCounter);
        if (Config.useSimpleStylometricFeatures) {
            this.lexicon.put("Number of lines", this.getNewFeatureId());
            this.lexicon.put("Average number of words per line", this.getNewFeatureId());
            this.lexicon.put("Number of unique words", this.getNewFeatureId());
            this.lexicon.put("Number of exotic words", this.getNewFeatureId());
            this.lexicon.put("Number of words", this.getNewFeatureId());
        }
        if (Config.useRhymeFeature) {
            this.lexicon.put("Rhyme", this.getNewFeatureId());
        }
        if (Config.useWordEmbeddings || Config.useNounEmbeddings) {
            // this.initialiseWordVectors("vec_songs_train.txt");
            this.initialiseWordVectors("GoogleNews-vectors-negative300.10000.txt");
            // this.initialiseWordVectors("vec_songs_dev_minimized.txt");
        }
        if (Config.useNounCountVectors || Config.useNounEmbeddings || Config.usePOSCountVectors) {
            // openNLP used for predicate extraction
            this.pos_model = new opennlp.tools.postag.POSModel(new File("en-pos-maxent.bin"));
            this.pos_tagger = new opennlp.tools.postag.POSTaggerME(pos_model);
        }
    }

    /**
     * Reading a file with word vectors and initialise the vectors into a mapping.
     * These vectors are used as representations of the words in the texts to
     * generalise better
     *
     * @param vectorFile
     * @throws IOException
     */
    private void initialiseWordVectors(String vectorFile) throws IOException {
        File fileToRead = new File(vectorFile);
        List<String> lines = Files.readAllLines(Paths.get(fileToRead.getAbsolutePath()));
        lines.remove(0);
        wordVectors = new LinkedHashMap<String, SparseVector>();
        int newFeatureID = this.getNewFeatureId();
        for (String line : lines) {
            String[] values = line.split("\\s+");
            SparseVector wordVector = new SparseVector();
            for (int i = 1; i < values.length; ++i) {
                wordVector.addFeature(newFeatureID + i - 1);
                wordVector.setFeatureValue(newFeatureID + i - 1, Double.parseDouble(values[i]));
            }
            wordVectors.put(values[0], wordVector);
        }
        this.featureIDCounter += Config.WORD_VEC_DIMENSIONS - 1;
    }

    public int getFeatureIDCounter() {
        return featureIDCounter;
    }

    public void setFeatureIDCounter(int featureIDCounter) {
        this.featureIDCounter = featureIDCounter;
    }

    @Override
    public FeatureVectorInterface createFeatureVector(Sample sample) {
        FeatureVector newFeatureVector = new FeatureVector();
        newFeatureVector.setGoldLabel(sample.getLabel());

        // add bias
        newFeatureVector.addFeature(this.lexicon.get("BIAS"));
        newFeatureVector.setFeatureValue(this.lexicon.get("BIAS"), 1.0);

        // add some counts
        if (Config.useSimpleStylometricFeatures) {
            newFeatureVector.addFeature(this.lexicon.get("Number of lines"));
            newFeatureVector.setFeatureValue(this.lexicon.get("Number of lines"),
                    this.extractNumberOfLines(sample.getText()));
            newFeatureVector.addFeature(this.lexicon.get("Average number of words per line"));
            newFeatureVector.setFeatureValue(this.lexicon.get("Average number of words per line"),
                    this.extractAverageNumberOfWordsPerLine(sample.getText()));
            newFeatureVector.addFeature(this.lexicon.get("Number of unique words"));
            newFeatureVector.setFeatureValue(this.lexicon.get("Number of unique words"),
                    this.extractNumberOfUniqueWords(sample.getText()));
            newFeatureVector.addFeature(this.lexicon.get("Number of exotic words"));
            newFeatureVector.setFeatureValue(this.lexicon.get("Number of exotic words"),
                    this.extractNumberOfExoticWords(sample.getText()));
            newFeatureVector.addFeature(this.lexicon.get("Number of words"));
            newFeatureVector.setFeatureValue(this.lexicon.get("Number of words"),
                    this.extractNumberOfWords(sample.getText()));
        }

        // add stylometric rhyme count
        if (Config.useRhymeFeature) {
            newFeatureVector.addFeature(this.lexicon.get("Rhyme"));
            newFeatureVector.setFeatureValue(this.lexicon.get("Rhyme"),
                    this.extractNormalisedNumberOfRhymes(sample.getText()));
        }

        // add word count vectors
        if (Config.useWordCountVectors && !Config.useNounCountVectors) {
            LinkedHashMap<Integer, Double> wordCountVector = this.createWordCountVector(sample.getText(), true);
            for (int key : wordCountVector.keySet()) {
                newFeatureVector.addFeature(key);
                newFeatureVector.setFeatureValue(key, wordCountVector.get(key));
            }
        }

        // add word count vectors for nouns only
        if (Config.useNounCountVectors && !Config.useWordCountVectors) {
            LinkedHashMap<Integer, Double> nounCountVector = this.extractNounCountVector(sample.getText());
            for (int key : nounCountVector.keySet()) {
                newFeatureVector.addFeature(key);
                newFeatureVector.setFeatureValue(key, nounCountVector.get(key));
            }
        }

        // add POS count vectors
        if (Config.usePOSCountVectors) {
            LinkedHashMap<Integer, Double> POSCountVector = this.extractPOSCountVector(sample.getText());
            // for (int key : countVector.keySet()) {
            // System.out.println(key + " " + countVector.get(key));
            // }
            // System.out.println("\n\n\n");
            // for (String word : this.lexicon.keySet()){
            // System.out.println(word + " " + this.lexicon.get(word));
            // } AAHHHRRRGGGGHHHHHHHHHHHHH
            for (int key : POSCountVector.keySet()) {
                // System.out.println(key);
                newFeatureVector.addFeature(key);
                newFeatureVector.setFeatureValue(key, POSCountVector.get(key));
            }
        }

        // add word count vectors fon nouns only
        if (Config.useNounEmbeddings && !Config.useWordEmbeddings) {
            newFeatureVector.vector = newFeatureVector.addVector(extractNounEmbeddingVector(sample.getText())).vector;
        }

        // adds the document vector to the the feature vector. IDs are taken care of
        if (Config.useWordEmbeddings) {
            newFeatureVector.vector = newFeatureVector.addVector(computeTextVector(sample.getText(), true)).vector;
        }
        return newFeatureVector;
    }

    /**
     * check for the last two letters in each line and see how many different ones
     * there are
     *
     * @param text the songtext
     * @return the amount of different line endings normalised by the length of the
     *         text
     */
    private Double extractNormalisedNumberOfRhymes(String text) {
        HashSet<String> lineEndings = new HashSet<>();
        for (String line : text.split("\n")) {
            String temp = line.replace("NEWLINE", "");
            temp = temp.replace(" ", "");
            temp = temp.replace(".", "");
            temp = temp.replace(")", "");
            temp = temp.replace("?", "");
            temp = temp.replace("!", "");
            temp = temp.replace(":", "");
            temp = temp.replace("]", "");
            // System.out.println(temp.subSequence(temp.length() - 2,
            // temp.length()).toString());
            lineEndings.add(temp.subSequence(temp.length() - 2, temp.length()).toString());
        }
        return lineEndings.size() / this.extractNumberOfLines(text);
    }

    /**
     * Method to find all nouns and represent them as count vectors
     *
     * @param text text to be extracted from
     * @return a count vector for nouns
     */
    private LinkedHashMap<Integer, Double> extractNounCountVector(String text) {
        StringBuilder nouns = new StringBuilder();
        for (String sentence : text.split("NEWLINE")) {
            String[] word_sequence = this.tokenise(this.preprocessSongText(sentence)).toArray(new String[0]);
            String[] tag_sequence = pos_tagger.tag(word_sequence);
            for (int i = 0; i < word_sequence.length; i++) {
                if (tag_sequence[i].equals("NN") || tag_sequence[i].equals("NNS") || tag_sequence[i].equals("NNP")) {
                    nouns.append(" NOUN_FEATURE_").append(word_sequence[i]);
                    // System.out.println(word_sequence[i] + " - " + tag_sequence[i]);
                }
            }
        }
        return createWordCountVector(nouns.toString(), false);
    }

    /**
     * Method get a representation of tall POS tags in a sequence as count vectors
     *
     * @param text text to be extracted from
     * @return a count vector for POS
     */
    private LinkedHashMap<Integer, Double> extractPOSCountVector(String text) {
        StringBuilder POS = new StringBuilder();
        for (String sentence : text.split("NEWLINE")) {
            String[] word_sequence = this.tokenise(this.preprocessSongText(sentence)).toArray(new String[0]);
            String[] tag_sequence = pos_tagger.tag(word_sequence);
            for (String tag : tag_sequence) {
                POS.append(" POS_FEATURE_").append(tag);
            }
        }
        // System.out.println(POS.toString());
        return createWordCountVector(POS.toString(), false);
    }

    /**
     * Method to find all nouns and represent them as embeddings
     *
     * @param text text to be extracted from
     * @return a count vector for nouns
     */
    private SparseVector extractNounEmbeddingVector(String text) {
        StringBuilder nouns = new StringBuilder();
        for (String sentence : text.split("NEWLINE")) {
            String[] word_sequence = this.tokenise(this.preprocessSongText(sentence)).toArray(new String[0]);
            String[] tag_sequence = pos_tagger.tag(word_sequence);
            for (int i = 0; i < word_sequence.length; i++) {
                if (tag_sequence[i].equals("NN") || tag_sequence[i].equals("NNS") || tag_sequence[i].equals("NNP")) {
                    nouns.append(" NOUN_FEATURE_").append(word_sequence[i]);
                    // System.out.println(word_sequence[i] + " - " + tag_sequence[i]);
                }
            }
        }
        return computeTextVector(nouns.toString(), true);
    }

    @Override
    public int getNewFeatureId() {
        this.featureIDCounter++;
        return this.featureIDCounter;
    }

    @Override
    public int lookupWordID(String word) {
        return this.lexicon.get(word);
    }

    @Override
    public LinkedHashMap<String, Integer> getLexicon() {
        return this.lexicon;
    }

    public void setLexicon(LinkedHashMap<String, Integer> lexicon) {
        this.lexicon = lexicon;
    }

    @Override
    public void addToLexicon(String word, int ID) {
        this.lexicon.put(word, ID);
    }

    @Override
    public LinkedHashMap<Integer, Double> createWordCountVector(String text, boolean preprocess) {
        LinkedHashMap<Integer, Double> wordCountVector = new LinkedHashMap<>();
        String words = text;

        // Preprocessing
        if (preprocess) {
            words = this.preprocessSongText(text);
            words = words.toLowerCase();
        }

        // Tokenisation
        for (String word : words.split("\\s+")) {
            if (!word.equals("")) {
                // System.out.println(word);
                if (!this.lexicon.containsKey(word)) {
                    // enter this branch if we encounter this word for the first
                    // time ever
                    this.lexicon.put(word, this.getNewFeatureId());
                    wordCountVector.put(lookupWordID(word), 1.0);
                } else if (!wordCountVector.containsKey(lookupWordID(word))) {
                    // enter this branch if we have seen this word before, but not
                    // in this sample
                    wordCountVector.put(lookupWordID(word), 1.0);
                } else {
                    // enter this branch if word has occurred in this sample before
                    double previousValue = wordCountVector.get(lookupWordID(word));
                    wordCountVector.put(lookupWordID(word), previousValue + 1.0);
                }
            }
        }
        if (Config.tfWeightingForCountVectors) {
            double wordCount = extractNumberOfWords(text);
            for (int dimension : wordCountVector.keySet()) {
                // apply tf weighting
                double tfCount = wordCountVector.get(dimension) / wordCount;
                wordCountVector.put(dimension, tfCount);
            }
        }
        return wordCountVector;
    }

    /**
     * Takes a songtext and cleans it from all noise
     *
     * @param text The text to be cleaned
     * @return The cleaned text
     */
    private String preprocessSongText(String text) {
        // array for replacements
        String[][] replacements = { { "[", " " }, { "]", " " }, { "NEWLINE", " " }, { "(", " " }, { ")", " " },
                { "Chorus", " " }, { "CHORUS", " " }, { "Verse", " " }, { "VERSE", " " }, { ".", " " }, { ":", " " },
                { ",", " " }, { ";", " " }, { "*", " " }, { "-", " " }, { "!", " " }, { "?", " " }, { "\"", " " },
                { "#", " " }, { "/", " " }, { "!", " " }, { "'", " '" } };
        // loop over the array and replace
        String words = text;
        for (String[] replacement : replacements) {
            words = words.replace(replacement[0], replacement[1]);
        }
        return words;
    }

    /**
     * Replaces long sequences of the same characters as in 'yeeaaaaaaaaahhh' with
     * shorter more unified ones
     *
     * @param tokens Takes a list of tokens to postptocess
     * @return The processed list of tokens
     */
    private List<String> postprocessSongText(List<String> tokens) {
        List<String> tokens2 = new ArrayList<String>();
        for (String token : tokens) {
            for (char c = 'a'; c <= 'z'; ++c) {
                String s = Character.toString(c);
                token = token.replaceAll(s + s + s + s + "+", s + s + s);
            }
            tokens2.add(token);
        }
        return tokens2;
    }

    /**
     * Reduces feature vectors according to a threshold and two boolean options.
     *
     * @param sampleFeatureVectorMapping The old feature vectors
     * @param threshold                  a threshold
     * @param count                      if true: the sum of occurrences is
     *                                   calculated for each feature; if false: the
     *                                   sum of values is calculated for each
     *                                   feature.
     * @param top                        if true: the features with sum >= the
     *                                   "threshold"-th greatest sum are kept; if
     *                                   false: the features with sum >= "threshold"
     *                                   are kept
     * @return The reduced feature vectors
     */
    private LinkedHashMap<Sample, FeatureVectorInterface> selectFrequentFeatures(
            LinkedHashMap<Sample, FeatureVectorInterface> sampleFeatureVectorMapping, int threshold, boolean count,
            boolean top) {
        int[] featureFreqs = new int[this.featureIDCounter + 1];

        for (FeatureVectorInterface featureVector : sampleFeatureVectorMapping.values()) {
            for (int feature : featureVector.getFeatureValues().keySet()) {
                Double a = featureVector.getFeatureValues().get(feature);
                if (count && a != 0) {
                    featureFreqs[feature] += 1;
                } else {
                    featureFreqs[feature] += a;
                }
            }
        }

        if (top) {
            int[] maxFreqs = featureFreqs.clone();
            Arrays.sort(maxFreqs);
            threshold = maxFreqs[maxFreqs.length - Math.min(threshold, maxFreqs.length)];
        }

        LinkedHashMap<Sample, FeatureVectorInterface> newSampleFeatureVectorMapping = new LinkedHashMap<Sample, FeatureVectorInterface>();

        for (Sample sample : sampleFeatureVectorMapping.keySet()) {
            FeatureVectorInterface newFeatureVector = new FeatureVector();
            for (int feature : sampleFeatureVectorMapping.get(sample).getFeatureValues().keySet()) {
                if (featureFreqs[feature] >= threshold) {
                    newFeatureVector.addFeature(feature);
                    newFeatureVector.setFeatureValue(feature,
                            sampleFeatureVectorMapping.get(sample).getFeatureValues().get(feature));
                }
            }
            newSampleFeatureVectorMapping.put(sample, newFeatureVector);
        }

        return newSampleFeatureVectorMapping;

    }

    @Override
    public LinkedHashMap<Sample, FeatureVectorInterface> onlyFrequentFeatures(
            LinkedHashMap<Sample, FeatureVectorInterface> sampleFeatureVectorMapping, int minFreq) {
        return selectFrequentFeatures(sampleFeatureVectorMapping, minFreq, true, false);
    }

    @Override
    public LinkedHashMap<Sample, FeatureVectorInterface> topFrequentFeatures(
            LinkedHashMap<Sample, FeatureVectorInterface> sampleFeatureVectorMapping, int topN) {
        return selectFrequentFeatures(sampleFeatureVectorMapping, topN, true, true);
    }

    @Override
    public LinkedHashMap<Sample, FeatureVectorInterface> removeHapaxLegomena(
            LinkedHashMap<Sample, FeatureVectorInterface> sampleFeatureVectorMapping, int minFreq) {
        return selectFrequentFeatures(sampleFeatureVectorMapping, minFreq, false, false);

    }

    @Override
    public LinkedHashMap<Sample, FeatureVectorInterface> removeHapaxLegomena(
            LinkedHashMap<Sample, FeatureVectorInterface> sampleFeatureVectorMapping) {
        return this.removeHapaxLegomena(sampleFeatureVectorMapping, 2);
    }

    public double extractNumberOfLines(String text) {
        if (text.equals("")) {
            return 0.0;
        }
        double lineCount = 1.0;
        for (String word : text.split("\\s+")) {
            if (word.equals("NEWLINE")) {
                lineCount++;
            }
        }
        return lineCount;
    }

    /**
     * Extracts the average number of words per line
     *
     * @param text The text to extract this feature from
     * @return The average number of words per line
     */
    public double extractAverageNumberOfWordsPerLine(String text) {
        if (text.equals("")) {
            return 0.0;
        }
        double length = 0.0;
        for (String word : text.split("\\s+")) {
            if (!word.equals("NEWLINE")) {
                length++;
            }
        }
        return length / this.extractNumberOfLines(text);
    }

    /**
     * Extracts the number of words that occur exactly once in a text
     *
     * @param text The text that this feature is extracted from
     * @return The number of words that occur exactly once in a text
     */
    public double extractNumberOfExoticWords(String text) {
        if (text.equals("")) {
            return 0.0;
        }
        HashSet<String> uniqueWords = new HashSet<>();
        HashSet<String> removedWords = new HashSet<>();
        for (String word : text.split("\\s+")) {
            if (!word.equals("NEWLINE")) {
                if (uniqueWords.contains(word)) {
                    uniqueWords.remove(word);
                    removedWords.add(word);
                } else {
                    if (!removedWords.contains(word)) {
                        uniqueWords.add(word);
                    }
                }
            }
        }
        return (double) uniqueWords.size();
    }

    /**
     * Extract the number of word types that occur in a text
     *
     * @param text The text that this feature is to be extracted from
     * @return The number of word types that occur in a text
     */
    public double extractNumberOfUniqueWords(String text) {
        if (text.equals("")) {
            return 0.0;
        }
        HashSet<String> uniqueWords = new HashSet<>();
        for (String word : text.split("\\s+")) {
            if (!word.equals("NEWLINE")) {
                uniqueWords.add(word);
            }
        }
        return (double) uniqueWords.size();
    }

    /**
     * Extract the number of tokens from a text
     *
     * @param text The text that feature is to be extracted from
     * @return The number of tokens
     */
    public double extractNumberOfWords(String text) {
        if (text.equals("")) {
            return 0.0;
        }
        ArrayList<String> words = new ArrayList<>();
        for (String word : text.split("\\s+")) {
            if (!word.equals("NEWLINE")) {
                words.add(word);
            }
        }
        return (double) words.size();
    }

    /**
     * Tokenise the songtext into a list of strings
     *
     * @param text The songtext
     * @return A list of tokens
     */
    public List<String> tokenise(String text) {
        text = preprocessSongText(text);
        StringTokenizer st = new StringTokenizer(text);
        List<String> tokens = new ArrayList<String>();
        while (st.hasMoreTokens()) {
            String token = st.nextToken();
            token = token.substring(0, 1) + (token.length() > 1 ? token.substring(1).toLowerCase() : "");
            tokens.add(token);
        }
        tokens = postprocessSongText(tokens);
        return tokens;
    }

    /**
     * Compute one vector for a document based on the vectors of the words it
     * contains
     *
     * @param text     The document we want to have the vector to
     * @param averaged Whether we want the vector to be averaged over all words to
     *                 get rid of the influence of the amount of words on the
     *                 document vector
     * @return A vector representing a document
     */
    public SparseVector computeTextVector(String text, boolean averaged) {
        List<String> tokens = tokenise(text);
        SparseVector averageVector = new SparseVector();
        for (String token : tokens) {
            averageVector = averageVector.addVector(wordVectors.getOrDefault(token, new SparseVector()));
        }
        if (averaged) {
            averageVector = averageVector.multiplyScalar(1.0 / tokens.size());
        }
        return averageVector;
    }

}
