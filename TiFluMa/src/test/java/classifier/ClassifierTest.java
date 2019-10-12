package classifier;

import evaluationMethods.Metrics;
import org.apache.uima.resource.ResourceInitializationException;
import org.junit.Ignore;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;

import static org.hamcrest.core.Is.is;
import static org.junit.Assert.assertThat;

public class ClassifierTest {

    @Test
    @Ignore
    public void playgroundTest() throws IOException, ResourceInitializationException {

        System.setProperty(org.slf4j.impl.SimpleLogger.DEFAULT_LOG_LEVEL_KEY, "DEBUG");
        ClassLoader loader = ClassLoader.getSystemClassLoader();

        Instant start = Instant.now();

        String trainFile = "songs_dev_minimized.txt";
        String testFile = "songs_dev_minimized.txt";

        // trains on big file
        ClassifierInterface cl = new Doc2Vec(
                new File(loader.getResource(trainFile).getFile()).getAbsolutePath(),
                new File(loader.getResource(testFile).getFile()).getAbsolutePath()
        );
        cl.initialiseClassifier();
        cl.trainClassifier(true);

        // evaluates on small file
        cl.runClassifier(new File(loader.getResource(testFile).getFile()));

        Metrics mt = new Metrics(
                new File(loader.getResource(testFile).getFile()).getAbsolutePath(),
                cl.getClassifierResults()
        );

        mt.readGoldFile();
        mt.evaluateSamples();
        System.out.println("Precision:              " + mt.retrieveMacroPrecision());
        System.out.println("Recall:                 " + mt.retrieveMacroRecall());
        System.out.println("Macro Averaged F-Score: " + mt.retrieveMacroFScore());
        System.out.println("Micro Averaged F-Score: " + mt.retrieveMicroFScore());

        Instant finish = Instant.now();

        long timeElapsed = Duration.between(start, finish).getSeconds();
        System.out.println(timeElapsed);
    }

    /**
     * Testing the perceptron with a small mock-up train file and a small
     * mock-up gold file
     *
     * @throws IOException
     */
    @Test
    @Ignore
    public void smallPerceptronTest() throws IOException {

        System.setProperty(org.slf4j.impl.SimpleLogger.DEFAULT_LOG_LEVEL_KEY, "DEBUG");
        ClassLoader loader = ClassLoader.getSystemClassLoader();

        Instant start = Instant.now();

        // trains on tiny file
        Classifier cl = new Classifier(new File(loader.getResource("mock_up_train.txt").getFile()).getAbsolutePath());
        cl.initialiseClassifier();
        cl.trainClassifier(true);

        // evaluates on tiny file
        cl.runClassifier(new File(loader.getResource("mock_up_test.txt").getFile()));

        Metrics mt = new Metrics(new File(loader.getResource("mock_up_test.txt").getFile()).getAbsolutePath(),
                cl.getClassifierResults());

        mt.readGoldFile();
        mt.evaluateSamples();
        System.out.println("Precision:              " + mt.getResultsAllClasses().precision());
        System.out.println("Recall:                 " + mt.getResultsAllClasses().recall());
        System.out.println("Macro Averaged F-Score: " + mt.retrieveMacroFScore());
        System.out.println("Micro Averaged F-Score: " + mt.retrieveMicroFScore());

        Instant finish = Instant.now();

        long timeElapsed = Duration.between(start, finish).getSeconds();
        System.out.println(timeElapsed);

        assertThat(mt.retrieveMacroFScore(), is(1.0));
        assertThat(mt.retrieveMicroFScore(), is(1.0));
    }
}
