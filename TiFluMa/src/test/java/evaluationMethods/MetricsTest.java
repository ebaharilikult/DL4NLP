package evaluationMethods;

import main.Config;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import static org.hamcrest.core.Is.is;
import static org.hamcrest.number.IsCloseTo.closeTo;
import static org.junit.Assert.assertThat;

public class MetricsTest {

    @BeforeClass
    public static void config() {
        Config.useRelevantArtistsOnly = false;
    }

    @Test
    public void testGoldFile() throws IOException {

        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(classLoader.getResource("test_gold_file.txt").getFile());

        // creating dummy results that mimic the classifiers output
        ArrayList<String> fake_results = new ArrayList<String>();
        fake_results.add("A");
        fake_results.add("A");
        fake_results.add("A");
        fake_results.add("C");

        // creating the metrics object used for evaluation
        Metrics m = new Metrics(file.getAbsolutePath(), fake_results);
        m.readGoldFile();
        m.evaluateSamples();

        // testing the F-Scores and Accuracies
        assertThat(m.retrieveMicroFScore(), is(0.5));
        assertThat(m.retrieveMacroFScore(), closeTo(0.26, 0.1));
        assertThat(m.retrieveAccuracy(), closeTo(0.66, 0.1));
    }

    @Test
    public void testEqalityForAllClasses() throws IOException {
        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(classLoader.getResource("test_gold_file.txt").getFile());

        // creating dummy results that mimic the classifiers output
        ArrayList<String> fake_results = new ArrayList<String>();
        fake_results.add("A");
        fake_results.add("A");
        fake_results.add("A");
        fake_results.add("A");

        // creating the metrics object used for evaluation
        Metrics m = new Metrics(file.getAbsolutePath(), fake_results);
        m.readGoldFile();
        m.evaluateSamples();

        // testing the F-Scores and Accuracies
        assertThat(m.retrieveMicroFScore(), is(0.5));
        assertThat(m.retrieveMacroFScore(), closeTo(0.26, 0.1));
        assertThat(m.retrieveAccuracy(), closeTo(0.5, 0.1));
    }

    @Test
    public void testEqualityForAllClasses() throws IOException {
        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(classLoader.getResource("test_gold_file.txt").getFile());

        // creating dummy results that mimic the classifiers output
        ArrayList<String> fake_results = new ArrayList<String>();
        fake_results.add("A");
        fake_results.add("A");
        fake_results.add("A");
        fake_results.add("A");

        // creating the metrics object used for evaluation
        Metrics m = new Metrics(file.getAbsolutePath(), fake_results);
        m.readGoldFile();
        m.evaluateSamples();

        // testing the F-Scores and Accuracies
        assertThat(m.retrieveMicroFScore(), is(0.5));
        assertThat(m.retrieveMacroFScore(), closeTo(0.26, 0.1));
        assertThat(m.retrieveAccuracy(), closeTo(0.5, 0.1));
    }

    @Test
    public void testInequalityForAllClasses() throws IOException {
        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(classLoader.getResource("test_gold_file.txt").getFile());

        // creating dummy results that mimic the classifiers output
        ArrayList<String> fake_results = new ArrayList<String>();
        fake_results.add("F");
        fake_results.add("F");
        fake_results.add("F");
        fake_results.add("F");

        // creating the metrics object used for evaluation
        Metrics m = new Metrics(file.getAbsolutePath(), fake_results);
        m.readGoldFile();
        m.evaluateSamples();

        // testing the F-Scores and Accuracies
        assertThat(m.retrieveMicroFScore(), is(0.0));
        assertThat(m.retrieveMacroFScore(), is(0.0));
        assertThat(m.retrieveAccuracy(), closeTo(0.3, 0.1));
    }
}
