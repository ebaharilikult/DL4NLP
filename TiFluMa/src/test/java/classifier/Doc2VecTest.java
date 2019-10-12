package classifier;

import evaluationMethods.Metrics;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.List;

public class Doc2VecTest {

    Logger logger = LoggerFactory.getLogger(Doc2VecTest.class);

    @Test
    @Ignore
    public void testDifferentLabels() throws Exception {
        ClassLoader loader = ClassLoader.getSystemClassLoader();
        String trainFile = "songs_dev_minimized.txt";
        String testFile = "songs_dev_minimized.txt";
        String fileToTrain = new File(loader.getResource(trainFile).getFile()).getAbsolutePath();
        String fileToTest = new File(loader.getResource(testFile).getFile()).getAbsolutePath();

        LabelAwareListSentenceIterator iterTrain = new LabelAwareListSentenceIterator(new FileInputStream(fileToTrain), "\t", 0, 2);
        int size = 0;
        while (iterTrain.hasNext()) {
            iterTrain.nextSentence();
            ++size;
        }
        iterTrain.reset();

        TokenizerFactory t = new UimaTokenizerFactory();

        ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
                .tokenizerFactory(t)
                .stopWords(new ArrayList<String>())
                .minWordFrequency(10)
                .windowSize(8)
                .layerSize(100)
                .batchSize(size)
                .learningRate(0.1)
                .minLearningRate(0.001)
                .iterate(iterTrain)
                .trainWordVectors(true)
                .epochs(10)
                .build();
        paragraphVectors.fit();

        LabelAwareListSentenceIterator iterTest = new LabelAwareListSentenceIterator(new FileInputStream(fileToTest), "\t", 0, 2);

        MeansBuilder meansBuilder = new MeansBuilder((InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(), t);
        LabelSeeker seeker = new LabelSeeker(paragraphVectors.getLabelsSource().getLabels(), (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

        ArrayList<String> classifierResults = new ArrayList<String>();
        List<String> goldLines = new ArrayList<String>();

        while (iterTest.hasNext()) {
            String sentence = iterTest.nextSentence();
            List<String> labels = iterTest.currentLabels();
            LabelledDocument document = new LabelledDocument();
            document.setContent(sentence);
            document.setLabels(labels);
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
            //logger.info(labels.get(0) + " is predicted as " + predictedLabel + ": " + maxScore);
            classifierResults.add(predictedLabel);
            goldLines.add(labels.get(0));

        }

        Metrics mt = new Metrics();
        mt.setClassifierResults(classifierResults);
        mt.setGoldLines(goldLines);
        mt.evaluateSamples();
        //System.out.println("Precision:              " + mt.retrieveMacroPrecision());
        //System.out.println("Recall:                 " + mt.retrieveMacroRecall());
        System.out.println("Macro Averaged F-Score: " + mt.retrieveMacroFScore());
        //System.out.println("Micro Averaged F-Score: " + mt.retrieveMicroFScore());


        /** Code examples:
         * https://deeplearning4j.org/docs/latest/deeplearning4j-nlp-doc2vec
         * https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/nlp/paragraphvectors/ParagraphVectorsClassifierExample.java
         */

    }

}
