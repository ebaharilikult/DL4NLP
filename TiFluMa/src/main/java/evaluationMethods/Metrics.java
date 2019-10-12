package evaluationMethods;

import classifier.Classifier;
import classifier.Sample;
import main.Config;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * Class used to store data for evaluation of the classifier and produce
 * evaluation metrics
 */
public class Metrics {

    Logger logger = LoggerFactory.getLogger(Metrics.class);

    private String pathToGoldFile;
    private ArrayList<String> classifierResults;
    private List<String> goldLines;
    private HashSet<String> allKnownLabels;
    private LinkedHashMap<String, EvaluationClassInstance> resultsPerClass = new LinkedHashMap<>();
    private EvaluationClassInstance resultsAllClasses = new EvaluationClassInstance();
    private LinkedHashMap<String, LinkedHashMap<String, Integer>> confusionmatrix = new LinkedHashMap<>();

    /**
     * constructor for the metrics object, used to evaluate the classifier
     *
     * @param pathToGoldFile    Path to a file containing the expected results, one
     *                          per line
     * @param classifierResults A list containing the results of the classifier
     */
    public Metrics(String pathToGoldFile, ArrayList<String> classifierResults) throws IOException {
        this.pathToGoldFile = pathToGoldFile;
        this.classifierResults = classifierResults;
        if (Config.fileIdentifier == null) {
            Config.fileIdentifier = java.time.LocalTime.now().toString().replace(":", "-").replace(".", "-");
        }
        if (Config.writeConfusionmatrixToFile) {
            FileWriter writer = new FileWriter("./src/main/python/Plotting/" + Config.userIntent.toString()
                    + "_confusionmatrix_of_classes_with_more_than_" + Config.artistsHaveToHaveMoreSongsThan
                    + "_samples_" + Config.fileIdentifier + ".csv", false);
            writer.write("");
            writer.close();
        }
        if (Config.writeResultsToFile) {
            FileWriter writer = new FileWriter("./src/main/python/Plotting/" + Config.userIntent.toString()
                    + "_results_with_more_than_" + Config.artistsHaveToHaveMoreSongsThan + "_samples_"
                    + java.time.LocalTime.now().toString().replace(":", "-").replace(".", "-") + ".csv", true);
            writer.write("Precision \t Recall \t Macro-f-score \t Micro-f-score\n");
            writer.close();
        }
    }

    public Metrics() {
    }

    public String getPathToGoldFile() {
        return pathToGoldFile;
    }

    public void setPathToGoldFile(String pathToGoldFile) {
        this.pathToGoldFile = pathToGoldFile;
    }

    public ArrayList<String> getClassifierResults() {
        return classifierResults;
    }

    public void setClassifierResults(ArrayList<String> classifierResults) {
        this.classifierResults = classifierResults;
    }

    public List<String> getGoldLines() {
        return goldLines;
    }

    public void setGoldLines(List<String> goldLines) {
        this.goldLines = goldLines;
    }

    public LinkedHashMap<String, EvaluationClassInstance> getResultsPerClass() {
        return resultsPerClass;
    }

    public void setResultsPerClass(LinkedHashMap<String, EvaluationClassInstance> resultsPerClass) {
        this.resultsPerClass = resultsPerClass;
    }

    public EvaluationClassInstance getResultsAllClasses() {
        return resultsAllClasses;
    }

    public void setResultsAllClasses(EvaluationClassInstance resultsAllClasses) {
        this.resultsAllClasses = resultsAllClasses;
    }

    /**
     * Fills the goldLines attribute with the expected results as strings
     *
     * @throws IOException
     */
    public void readGoldFile() throws IOException {
        if (Config.useRelevantArtistsOnly) {
            // Dirty workaround, ouch this hurts to look at, better don't even look at it
            // and move on
            Classifier cl = new Classifier(new File("/home/artist/songs_train.txt").getAbsolutePath(),
                    this.pathToGoldFile);
            // cl now has the relevant classes saved. This is nice.
            // now we want the gold lines to be the labels of the evaluation
            // data, that already throws out the little ones.
            for (Sample sample : cl.readSamplesFromFile(this.pathToGoldFile)) {
                if (this.goldLines != null) {
                    this.goldLines.add(sample.getLabel());
                } else {
                    this.goldLines = new ArrayList<String>();
                    this.goldLines.add(sample.getLabel());
                }

            }
        } else {
            setGoldLines(Files.readAllLines(Paths.get(getPathToGoldFile())));
            for (int i = 0; i < getGoldLines().size(); ++i) {
                getGoldLines().set(i, getGoldLines().get(i).split("\t")[0]);
            }
        }
    }

    /**
     * All labels that occur in the results and the expected results are being
     * collected into a set
     */
    private void createAllClasses() {
        this.allKnownLabels = new HashSet<>();
        this.allKnownLabels.addAll(getClassifierResults());
        for (String line : getGoldLines()) {
            this.allKnownLabels.add(line.trim());
        }
        for (String label : this.allKnownLabels) {
            this.confusionmatrix.put(label, new LinkedHashMap<String, Integer>());
        }
    }

    /**
     * Creating EvaluationClassInstance objects for every label that we have to deal
     * with and storing them in a mapping with their label as the key
     */
    private void initialiseResultsPerClassEntries() {
        for (String label : this.allKnownLabels) {
            this.resultsPerClass.put(label, new EvaluationClassInstance());
        }
    }

    /**
     * Main routine calling all subroutines needed to get all the TPs FPs TNs and
     * FNs for further use in micro and macro evaluations
     *
     * @throws IOException
     */
    public void evaluateSamples() throws IOException {
        createAllClasses();
        initialiseResultsPerClassEntries();
        Iterator<String> classifierResultIterator = classifierResults.iterator();
        for (String gold : getGoldLines()) {
            String prediction = classifierResultIterator.next();
            logger.debug("Gold: " + gold);
            logger.debug("Pred: " + prediction);
            evaluateSample(gold, prediction);
        }
        if (Config.writeConfusionmatrixToFile) {
            FileWriter writer = new FileWriter("./src/main/python/Plotting/" + Config.userIntent.toString()
                    + "_confusionmatrix_of_classes_with_more_than_" + Config.artistsHaveToHaveMoreSongsThan
                    + "_samples_" + Config.fileIdentifier + ".csv", false);
            writer.write(generateConfusionMatrixAsCSV());
            writer.close();
        }
        if (Config.writeResultsToFile) {
            FileWriter writer = new FileWriter("./src/main/python/Plotting/" + Config.userIntent.toString()
                    + "_results_with_more_than_" + Config.artistsHaveToHaveMoreSongsThan + "_samples_"
                    + java.time.LocalTime.now().toString().replace(":", "-").replace(".", "-") + ".csv", true);
            writer.write(this.retrieveMacroPrecision() + "\t" + this.retrieveMacroRecall() + "\t"
                    + this.retrieveMacroFScore() + "\t" + this.retrieveMicroFScore() + "\n");
            writer.close();
        }
    }

    /**
     * returns the confusionmatrix as CSV
     */
    private String generateConfusionMatrixAsCSV() {
        ArrayList<String> classes = new ArrayList<>(this.confusionmatrix.keySet());
        StringBuilder matrix = new StringBuilder();
        matrix.append("X\t");
        for (String labelCol : classes) {
            matrix.append(labelCol).append("\t");
        }
        matrix.append("\n");
        for (String labelCol : classes) {
            matrix.append(labelCol).append("\t");
            for (String labelRow : classes) {
                matrix.append(this.confusionmatrix.get(labelCol).getOrDefault(labelRow, 0)).append("\t");
            }
            matrix.append("\n");
        }
        return matrix.toString();
    }

    /**
     * Evaluating the result of one single instance that was classified
     *
     * @param gold       The expected result
     * @param prediction The result the classifier produced
     */
    private void evaluateSample(String gold, String prediction) {
        int prevValue = confusionmatrix.get(gold).getOrDefault(prediction, 0);
        confusionmatrix.get(gold).put(prediction, prevValue + 1);
        if (gold.equals(prediction)) {
            handleTruePredictions(gold);
        }
        if (!gold.equals(prediction)) {
            handleFalsePredictions(gold, prediction);
        }
    }

    /**
     * Evaluating the result of one single instance that was classified wrong
     *
     * @param gold       The expected result
     * @param prediction The result the classifier produced
     */
    private void handleFalsePredictions(String gold, String prediction) {
        this.resultsPerClass.get(gold).increaseFalseNegatives();
        this.resultsAllClasses.increaseFalseNegatives();
        handleFalsePositivesAndTrueNegatives(gold, prediction);
    }

    /**
     * Evaluating the result of one single instance that was classified wrong from
     * the viewpoint of the prediction and the gold label
     *
     * @param gold       The expected result
     * @param prediction The result the classifier produced
     */
    private void handleFalsePositivesAndTrueNegatives(String gold, String prediction) {
        for (String label : resultsPerClass.keySet()) {
            if (label.equals(prediction)) {
                this.resultsPerClass.get(label).increaseFalsePositives();
                this.resultsAllClasses.increaseFalsePositives();
            } else if (!label.equals(gold)) {
                this.resultsPerClass.get(label).increaseTrueNegatives();
                this.resultsAllClasses.increaseTrueNegatives();
            }
        }
    }

    /**
     * Handling the evaluation of a sample that was classified correctly
     *
     * @param gold the expected result
     */
    private void handleTruePredictions(String gold) {
        this.resultsPerClass.get(gold).increaseTruePositives();
        this.resultsAllClasses.increaseTruePositives();
        handleTrueNegatives(gold);
    }

    /**
     * Handling the evaluation of a sample that was classified correctly
     *
     * @param gold the expected result
     */
    private void handleTrueNegatives(String gold) {
        for (String label : resultsPerClass.keySet()) {
            if (!label.equals(gold)) {
                this.resultsPerClass.get(label).increaseTrueNegatives();
                this.resultsAllClasses.increaseTrueNegatives();
            }
        }
    }

    /**
     * Calculate the F-Score of all classes added together
     *
     * @return Micro Averaged F-Score
     */
    public double retrieveMicroFScore() {
        return this.resultsAllClasses.fScore();
    }

    /**
     * Calculate the F-Score of all the classes individually and then add them up
     * and get the average
     *
     * @return Macro Averaged F-Score
     */
    public double retrieveMacroFScore() {
        double macroFScore = 0.0;
        for (String label : this.allKnownLabels) {
            macroFScore += this.resultsPerClass.get(label).fScore();
        }
        macroFScore /= this.allKnownLabels.size();
        return macroFScore;
    }

    /**
     * Calculate the recall of all the classes individually and then add them up and
     * get the average
     *
     * @return Macro Averaged recall
     */
    public double retrieveMacroRecall() {
        double macroRecall = 0.0;
        for (String label : this.allKnownLabels) {
            macroRecall += this.resultsPerClass.get(label).recall();
        }
        macroRecall /= this.allKnownLabels.size();
        return macroRecall;
    }

    /**
     * Calculate the precision of all the classes individually and then add them up
     * and get the average
     *
     * @return Macro Averaged precision
     */
    public double retrieveMacroPrecision() {
        double macroPrecision = 0.0;
        for (String label : this.allKnownLabels) {
            macroPrecision += this.resultsPerClass.get(label).precision();
        }
        macroPrecision /= this.allKnownLabels.size();
        return macroPrecision;
    }

    /**
     * Calculate the Accuracy of all classes
     *
     * @return Micro Averaged Accuracy
     */
    public double retrieveAccuracy() {
        return this.resultsAllClasses.accuracy();
    }

}
