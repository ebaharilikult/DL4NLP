package classifier;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public interface ClassifierInterface {

    ArrayList<String> getClassifierResults();

    void setClassifierResults(List<String> list);

    /**
     * Classify all samples in the given file.
     *
     * @param fileToClassify
     * @throws IOException
     */
    void runClassifier(File fileToClassify) throws IOException;

    /**
     * Classify all samples from the evaluation file
     *
     * @throws IOException
     */
    void runClassifier() throws IOException;

    /**
     * Train perceptron on training file.
     *
     * @param evaluateAfterEachEpoch if the evaluation file should be predicted
     * @throws IOException
     */
    void trainClassifier(Boolean evaluateAfterEachEpoch) throws IOException;

    /**
     * Initialise classifier
     *
     * @throws IOException
     */
    void initialiseClassifier() throws IOException;

    /**
     * Reads the samples from the given file path.
     *
     * @param fileToRead path to read the samples from
     * @return List of Sample objects  (or Sentence iterator object) from file
     * @throws IOException if the file could not be read
     */
    Object readSamplesFromFile(String fileToRead) throws IOException;

}
