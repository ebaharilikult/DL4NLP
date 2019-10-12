package classifier;

import evaluationMethods.Metrics;
import main.Config;
import org.apache.uima.resource.ResourceInitializationException;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/**
 * Uses deeplearning4j's Doc2Vec implementation.
 */
public class Doc2Vec implements ClassifierInterface {

    Logger logger = LoggerFactory.getLogger(Doc2Vec.class);

    private ArrayList<String> classifierResults;
    private ParagraphVectors paragraphVectors;
    private String trainingFilePath;
    private String evaluationFilePath;
    private TokenizerFactory tokeniser;
    private MeansBuilder meansBuilder;
    private LabelSeeker seeker;
    private HashSet<String> relevantArtists = null;

    public Doc2Vec(String trainingFile) throws IOException, ResourceInitializationException {
        classifierResults = new ArrayList<>();
        this.trainingFilePath = trainingFile;
        this.tokeniser = new UimaTokenizerFactory();
        if (Config.useRelevantArtistsOnly) {
            this.relevantArtists = ClassifierUtils.extractRelevantArtists(trainingFilePath, evaluationFilePath);
        }
    }

    public Doc2Vec(String trainingFile, String evaluationFile) throws IOException, ResourceInitializationException {
        logger.debug("Classifier > trainingFile='{}', evaluationFile='{}'", trainingFile, evaluationFile);
        classifierResults = new ArrayList<>();
        this.trainingFilePath = trainingFile;
        this.evaluationFilePath = evaluationFile;
        this.tokeniser = new UimaTokenizerFactory();
        if (Config.useRelevantArtistsOnly) {
            this.relevantArtists = ClassifierUtils.extractRelevantArtists(trainingFilePath, evaluationFilePath);
        }
        logger.debug("Classifier <");
    }

    @Override
    public void runClassifier(File fileToClassify) throws IOException {
        logger.debug("runClassifier > fileToClassify='{}'", fileToClassify);
        LabelAwareListSentenceIterator iterTest = readSamplesFromFile(fileToClassify.getAbsolutePath()); // read
        // fileToClassify

        // runs perceptron and classify given samples into classifierResults
        setClassifierResults(this.predict(iterTest));
        logger.debug("runClassifier <");
    }

    @Override
    public void runClassifier() throws IOException {
        logger.debug("runClassifier >");
        LabelAwareListSentenceIterator iterTest = readSamplesFromFile(evaluationFilePath); // read
        // fileToClassify

        // runs perceptron and classify given samples into classifierResults
        setClassifierResults(this.predict(iterTest));
        logger.debug("runClassifier <");
    }

    /**
     * Predicts the label for the given text
     *
     * @param sentence text to predict
     * @return predicted label
     */
    private String predict(String sentence) {
        LabelledDocument document = new LabelledDocument();
        document.setContent(sentence);
        INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
        List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

        String predictedLabel = "";
        double maxScore = Double.NEGATIVE_INFINITY;
        for (Pair<String, Double> score : scores) {
            if (score.getSecond() > maxScore) {
                maxScore = score.getSecond();
                predictedLabel = score.getFirst();
            }
        }

        return predictedLabel;
    }

    /**
     * Predict a label for each document in an iterator and return the list of predicted
     * labels.
     *
     * @param iterTest
     * @return
     */
    private List<String> predict(LabelAwareListSentenceIterator iterTest) {
        ArrayList<String> classifierResults = new ArrayList<String>();

        while (iterTest.hasNext()) {
            String sentence = iterTest.nextSentence();
            classifierResults.add(predict(sentence));
        }

        iterTest.reset();
        return classifierResults;

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

    /**
     * Collects the labels of all documents of an iterator.
     *
     * @param iterTest
     * @return The gold labels in the same order.
     */
    private List<String> collectGoldLines(LabelAwareListSentenceIterator iterTest) {
        List<String> goldLines = new ArrayList<String>();
        while (iterTest.hasNext()) {
            iterTest.nextSentence();
            List<String> labels = iterTest.currentLabels();
            goldLines.add(labels.get(0));
        }
        iterTest.reset();
        return goldLines;
    }

    /**
     * Evaluate the samples given for intermediate results
     *
     * @param iterTest Evaluation data
     */
    private void evaluateSamples(LabelAwareListSentenceIterator iterTest) {
        ArrayList<String> evaluationResults = (ArrayList<String>) predict(iterTest);

        Metrics mt = new Metrics();
        mt.setClassifierResults(evaluationResults);
        mt.setGoldLines(collectGoldLines(iterTest));
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

    /**
     * Builds paragraph vectors with Doc2Vec's ParagraphVectors.Builder()
     * Uses parameters of Config.
     *
     * @param iterTrain Training samples as iterator.
     * @param epochs    No. of epochs to train.
     */
    private void buildParagraphVectors(LabelAwareListSentenceIterator iterTrain, int epochs) {
        paragraphVectors = new ParagraphVectors.Builder()
                .minWordFrequency(Config.MIN_WORD_FREQUENCY)
                .layerSize(Config.WORD_VEC_DIMENSIONS)
                .stopWords(new ArrayList<String>())
                .windowSize(Config.WINDOW_SIZE)
                .learningRate(Config.LEARNING_RATE)
                .minLearningRate(0.001)
                .batchSize(Math.min(Config.BATCH_SIZE, iterTrain.getText().size()))
                .epochs(epochs)
                .iterate(iterTrain)
                .trainWordVectors(true)
                .tokenizerFactory(tokeniser)
                .build();
        paragraphVectors.fit();
        meansBuilder = new MeansBuilder((InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(), tokeniser);
        seeker = new LabelSeeker(paragraphVectors.getLabelsSource().getLabels(), (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());
        iterTrain.reset();
    }

    @Override
    public void trainClassifier(Boolean evaluateAfterEachTenthEpoch) throws IOException {
        LabelAwareListSentenceIterator iterTrain = readSamplesFromFile(trainingFilePath);
        if (evaluateAfterEachTenthEpoch) {
            LabelAwareListSentenceIterator iterTest = readSamplesFromFile(evaluationFilePath);
            for (int i = 10; i < Config.EPOCHS; i += 10) {
                buildParagraphVectors(iterTrain, i);
                evaluateSamples(iterTest);
            }
        }
        buildParagraphVectors(iterTrain, Config.EPOCHS);
    }

    @Override
    public void initialiseClassifier() throws IOException {
    }

    @Override
    public LabelAwareListSentenceIterator readSamplesFromFile(String fileToRead) throws IOException {
        LabelAwareListSentenceIterator iterator = new LabelAwareListSentenceIterator(new FileInputStream(fileToRead), "\t", 0, 2);
        if (Config.useRelevantArtistsOnly) {
            if (this.relevantArtists != null) {
                List<String> relevantLabels = new ArrayList<>();
                List<String> relevantText = new ArrayList<>();
                for (int i = 0; i < iterator.getLabels().size(); ++i) {
                    if (this.relevantArtists.contains(iterator.getLabels().get(i))) {
                        relevantLabels.add(iterator.getLabels().get(i));
                        relevantText.add(iterator.getText().get(i));
                    }
                }
                iterator.setLabels(relevantLabels);
                iterator.setText(relevantText);
            }
        }
        return iterator;
    }
}
