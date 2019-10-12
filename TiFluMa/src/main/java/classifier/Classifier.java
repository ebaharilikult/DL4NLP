package classifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import main.Config;

/**
 * Uses a multi-class perceptron. Takes files to train the perceptron or to use
 * the perceptron to classify them.
 */
public class Classifier implements ClassifierInterface {

    Logger logger = LoggerFactory.getLogger(Classifier.class);
    FeatureExtractorInterface featureExtractor;
    private ArrayList<String> classifierResults;
    private PerceptronInterface perceptron;
    private String trainingFilePath;
    private String evaluationFilePath;
    private String testFilePath;
    private HashSet<String> relevantArtists = null;

    private int featureCounter = 0;

    public Classifier() {
    }

    public Classifier(String trainingFile) throws IOException {
        classifierResults = new ArrayList<>();
        this.trainingFilePath = trainingFile;
        if (Config.useRelevantArtistsOnly) {
            this.relevantArtists = ClassifierUtils.extractRelevantArtists(trainingFilePath, evaluationFilePath);
        }
        featureExtractor = new FeatureExtractor();
        this.perceptron = new Perceptron(featureExtractor);
    }

    public Classifier(String trainingFile, String evaluationFile) throws IOException {
        logger.debug("Classifier > trainingFile='{}', evaluationFile='{}'", trainingFile, evaluationFile);
        classifierResults = new ArrayList<>();
        this.trainingFilePath = trainingFile;
        this.evaluationFilePath = evaluationFile;
        if (Config.useRelevantArtistsOnly) {
            this.relevantArtists = ClassifierUtils.extractRelevantArtists(trainingFilePath, evaluationFilePath);
        }
        featureExtractor = new FeatureExtractor();
        this.perceptron = new Perceptron(featureExtractor);
        logger.debug("Classifier <");
    }

    public Classifier(String trainingFile, String evaluationFile, String testFile) throws IOException {
        logger.debug("Classifier > trainingFile='{}', evaluationFile='{}, testFile='{}'", trainingFile, evaluationFile,
                testFile);
        classifierResults = new ArrayList<>();
        this.trainingFilePath = trainingFile;
        this.evaluationFilePath = evaluationFile;
        this.testFilePath = testFile;
        if (Config.useRelevantArtistsOnly) {
            this.relevantArtists = ClassifierUtils.extractRelevantArtists(trainingFilePath, evaluationFilePath);
        }
        featureExtractor = new FeatureExtractor();
        this.perceptron = new Perceptron(featureExtractor);
        logger.debug("Classifier <");
    }

    public PerceptronInterface getPerceptron() {
        return perceptron;
    }

    /**
     * Produce a fake output (without file to classify).
     *
     * @throws IOException
     */
    public void runFakeClassifier() throws IOException {
        logger.debug("runFakeClassifier >");
        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(classLoader.getResource("songs_dev-predicted.txt").getFile());
        setClassifierResults(new ArrayList<String>(Files.readAllLines(file.toPath())));
        for (int i = 0; i < getClassifierResults().size(); ++i) {
            getClassifierResults().set(i, getClassifierResults().get(i).split("\t")[0]);
        }
        logger.debug("runFakeClassifier <");
    }

    @Override
    public void runClassifier(File fileToClassify) throws IOException {
        logger.debug("runClassifier > fileToClassify='{}'", fileToClassify);
        List<Sample> samples = readSamplesFromFile(fileToClassify.getAbsolutePath()); // read
        // fileToClassify

        // runs perceptron and classify given samples into classifierResults
        setClassifierResults(perceptron.predict(samples));
        logger.debug("runClassifier <");
    }

    @Override
    public void runClassifier() throws IOException {
        logger.debug("runClassifier >");
        List<Sample> samples = readSamplesFromFile(evaluationFilePath); // read
        // fileToClassify

        // runs perceptron and classify given samples into classifierResults
        setClassifierResults(perceptron.predict(samples));
        logger.debug("runClassifier <");
    }

    @Override
    public ArrayList<String> getClassifierResults() {
        if (this.classifierResults == null)
            this.classifierResults = new ArrayList<String>();
        return this.classifierResults;
    }

    @Override
    public void setClassifierResults(List<String> list) {
        this.classifierResults.clear();
        this.classifierResults.addAll(list);
    }

    @Override
    public void trainClassifier(Boolean evaluateAfterEachEpoch) throws IOException {
        List<Sample> evaluationSamples = new ArrayList<Sample>();
        if (!StringUtils.isEmpty(evaluationFilePath))
            evaluationSamples = readSamplesFromFile(evaluationFilePath);

        perceptron.train(evaluationSamples, evaluateAfterEachEpoch);
    }

    @Override
    public void initialiseClassifier() throws IOException {
        List<Sample> samples = readSamplesFromFile(trainingFilePath);

        LinkedHashMap<Sample, FeatureVectorInterface> features = extractFeatures(samples);
        perceptron.initialise(features);
    }

    private LinkedHashMap<Sample, FeatureVectorInterface> extractFeatures(List<Sample> samples) {
        LinkedHashMap<Sample, FeatureVectorInterface> features = new LinkedHashMap<Sample, FeatureVectorInterface>(
                8192);

        for (Sample sample : samples) {
            FeatureVectorInterface featureVector = this.featureExtractor.createFeatureVector(sample);
            features.put(sample, featureVector);
        }

        // remove uninfomative features
        features = this.featureExtractor.onlyFrequentFeatures(features, 10);
        // sampleFeatureVectorMapping =
        // this.featureExtractor.removeHapaxLegomena(sampleFeatureVectorMapping,
        // 10);
        // sampleFeatureVectorMapping =
        // this.featureExtractor.topFrequentFeatures(sampleFeatureVectorMapping,
        // 10);

        return features;
    }

    /**
     * Extracts the train features to the given file path as a compressed file.
     *
     * @param exportPath path to export the features
     * @throws IOException
     */
    public void extractTrainFeaturesToFile(String exportPath) throws IOException {
        List<Sample> trainSamples = readSamplesFromFile(trainingFilePath);
        extractFeaturesToFile(trainSamples, exportPath, true);
    }

    /**
     * Extracts the evaluation features to the given file path as a compressed file.
     *
     * @param exportPath path to export the features
     * @throws IOException
     */
    public void extractEvaluationFeaturesToFile(String exportPath) throws IOException {
        List<Sample> evalSamples = readSamplesFromFile(evaluationFilePath);
        extractFeaturesToFile(evalSamples, exportPath, false);
    }

    /**
     * Extracts the test features to the given file path as a compressed file.
     *
     * @param exportPath path to export the features
     * @throws IOException
     */
    public void extractTestFeaturesToFile(String exportPath) throws IOException {
        List<Sample> evalSamples = readSamplesFromFile(testFilePath);
        extractFeaturesToFile(evalSamples, exportPath, false);
    }

    /**
     * @param samples
     * @param exportPath
     * @throws IOException
     */
    private void extractFeaturesToFile(List<Sample> samples, String exportPath, Boolean saveFeatureCount)
            throws IOException {
        LinkedHashMap<Sample, FeatureVectorInterface> features = extractFeatures(samples);
        if (saveFeatureCount) {
            this.featureCounter = this.featureExtractor.getFeatureIDCounter();
        }
        exportFeatueVectorsToCSV(features, exportPath);

    }

    /**
     * Exports the features as a compressed zip file
     *
     * @param features   features to export
     * @param exportPath path
     * @throws IOException
     */
    private void exportFeatueVectorsToCSV(LinkedHashMap<Sample, FeatureVectorInterface> features, String exportPath)
            throws IOException {

        OutputStream outStream = new FileOutputStream(exportPath);
        ZipOutputStream zipOut = new ZipOutputStream(outStream);
        final OutputStreamWriter osw = new OutputStreamWriter(zipOut);
        CSVPrinter printer = new CSVPrinter(osw, CSVFormat.TDF);

        // only one csv entry right now
        if (features != null) {
            zipOut.putNextEntry(new ZipEntry("data.csv"));
            printCSVEntry(features, printer);
        }

        printer.close();
        zipOut.close();
        outStream.close();

    }

    /**
     * Prints given features against an already defined csv entry
     *
     * @param features features to print
     * @param printer  contains the current csv entry
     */
    private void printCSVEntry(LinkedHashMap<Sample, FeatureVectorInterface> features, CSVPrinter printer) {

        List<String> headerEntry = generateCSVHeader();
        printListToCSV(headerEntry, printer);

        features.forEach((sample, featureVector) -> {
            List<String> csvEntryList = generateCSVFeatureEntry(sample.getLabel(), featureVector);
            printListToCSV(csvEntryList, printer);
        });
    }

    /**
     * Creates and returns an csv row with all the given features
     *
     * @param sampleLabel label to set in the first column
     * @param v           feature vector
     * @return list that contains the csv row
     */
    private List<String> generateCSVFeatureEntry(String sampleLabel, FeatureVectorInterface v) {
        List<String> csvEntryList = new ArrayList<>();
        csvEntryList.add(sampleLabel);

        for (int i = 0; i <= this.featureCounter; i++) {
            if (v.getFeatureValue(i) != null)
                csvEntryList.add(String.format(Locale.ROOT, "%.4f", v.getFeatureValue(i)));
            else
                csvEntryList.add("0");
        }
        return csvEntryList;
    }

    /**
     * Creates and returns a list which contains the csv header row
     *
     * @return
     */
    private List<String> generateCSVHeader() {
        List<String> csvEntryList = new ArrayList<>();
        csvEntryList.add("Label");
        for (int i = 0; i <= this.featureCounter; i++) {
            csvEntryList.add(String.valueOf(i));
        }
        return csvEntryList;
    }

    /**
     * Prints the given csv row to the csv file
     *
     * @param csvEntryList csv row as a list
     * @param printer      printer to print the csv
     */
    private void printListToCSV(List<String> csvEntryList, CSVPrinter printer) {
        try {
            printer.printRecord(csvEntryList);
        } catch (IOException e) {
            // just log
            logger.error("Failed to print csv entry", e);
        }
    }

    @Override
    public List<Sample> readSamplesFromFile(String fileToRead) throws IOException {
        List<Sample> samples = new ArrayList<>();

        samples.addAll(Files.lines(Paths.get(fileToRead)).map(line -> line.split("\t"))
                .map(splitted -> new Sample(splitted[0], splitted[1], splitted[2])).collect(Collectors.toList()));

        if (Config.useRelevantArtistsOnly) {
            if (this.relevantArtists != null) {
                List<Sample> relevantSamples = new ArrayList<>();
                for (Sample sample : samples) {
                    if (this.relevantArtists.contains(sample.getLabel())) {
                        relevantSamples.add(sample);
                    }
                }
                return relevantSamples;
            }
        }
        return samples;
    }
}
