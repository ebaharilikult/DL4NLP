package classifier;

import main.Config;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

import static org.hamcrest.core.Is.is;
import static org.junit.Assert.assertThat;

public class FeatureExtractorTest {

    @Test
    public void testWordCountVector() throws IOException {
        Config.useSimpleStylometricFeatures = false;
        Config.useRhymeFeature = false;
        Config.useNounCountVectors = false;
        Config.useWordCountVectors = true;
        Config.usePOSCountVectors = false;
        Config.tfWeightingForCountVectors = false;
        Config.useNounEmbeddings = false;
        Config.useWordEmbeddings = false;
        FeatureExtractor fe = new FeatureExtractor();
        LinkedHashMap<Integer, Double> we = fe.createWordCountVector("a a a a b b c", false);
        assertThat(we.get(1), is(4.0));
        assertThat(we.get(2), is(2.0));
        assertThat(we.get(3), is(1.0));
    }

    @Test
    public void testExtractNumberOfLines() throws IOException {
        FeatureExtractor fe = new FeatureExtractor();
        assertThat(fe.extractNumberOfLines("a NEWLINE b NEWLINE c"), is(3.0));
        assertThat(fe.extractNumberOfLines("NEWLINE"), is(2.0));
        assertThat(fe.extractNumberOfLines(""), is(0.0));
    }

    @Test
    public void testExtractAverageNumberOfWordsPerLine() throws IOException {
        FeatureExtractor fe = new FeatureExtractor();
        assertThat(fe.extractAverageNumberOfWordsPerLine("a NEWLINE b NEWLINE c"), is(1.0));
        assertThat(fe.extractAverageNumberOfWordsPerLine("NEWLINE"), is(0.0));
        assertThat(fe.extractAverageNumberOfWordsPerLine(""), is(0.0));
        assertThat(fe.extractAverageNumberOfWordsPerLine("a a a a NEWLINE a a"), is(3.0));
    }

    @Test
    public void testExtractNumberOfUniqueWords() throws IOException {
        FeatureExtractor fe = new FeatureExtractor();
        assertThat(fe.extractNumberOfUniqueWords("a NEWLINE b NEWLINE c"), is(3.0));
        assertThat(fe.extractNumberOfUniqueWords("NEWLINE"), is(0.0));
        assertThat(fe.extractNumberOfUniqueWords(""), is(0.0));
        assertThat(fe.extractNumberOfUniqueWords("a a a a NEWLINE a a"), is(1.0));
    }

    @Test
    public void testExtractNumberOfExoticWords() throws IOException {
        FeatureExtractor fe = new FeatureExtractor();
        assertThat(fe.extractNumberOfExoticWords("a NEWLINE b NEWLINE c"), is(3.0));
        assertThat(fe.extractNumberOfExoticWords("NEWLINE"), is(0.0));
        assertThat(fe.extractNumberOfExoticWords(""), is(0.0));
        assertThat(fe.extractNumberOfExoticWords("a a a a NEWLINE a a"), is(0.0));
    }

    @Test
    public void testExtractNumberOfWords() throws IOException {
        FeatureExtractor fe = new FeatureExtractor();
        assertThat(fe.extractNumberOfWords("a NEWLINE b NEWLINE c"), is(3.0));
        assertThat(fe.extractNumberOfWords("NEWLINE"), is(0.0));
        assertThat(fe.extractNumberOfWords(""), is(0.0));
        assertThat(fe.extractNumberOfWords("a a a a NEWLINE a a"), is(6.0));
    }

    @Test
    public void testSelectFrequentFeatures() throws IOException {
        Config.useNounCountVectors = false;
        Config.useNounEmbeddings = false;
        Config.useWordCountVectors = true;
        Config.useSimpleStylometricFeatures = false;
        Config.useWordEmbeddings = false;
        Config.tfWeightingForCountVectors = false;

        Sample a = new Sample("A", "A", "this is a test, a test");
        Sample b = new Sample("B", "B", "this is another test with lots of unique words");
        Sample c = new Sample("C", "C", "yet another test!");

        List<Sample> samples = new ArrayList<Sample>();
        samples.add(a);
        samples.add(b);
        samples.add(c);

        FeatureExtractor fe = new FeatureExtractor();

        LinkedHashMap<Sample, FeatureVectorInterface> sampleFeatureVectorMapping = new LinkedHashMap<Sample, FeatureVectorInterface>();
        for (Sample sample : samples) {
            sampleFeatureVectorMapping.put(sample, fe.createFeatureVector(sample));
        }

        LinkedHashMap<String, Integer> lexicon = fe.getLexicon();
        System.out.println(lexicon.keySet());

        LinkedHashMap<Sample, FeatureVectorInterface> newSampleFeatureVectorMapping;

        newSampleFeatureVectorMapping = fe.removeHapaxLegomena(sampleFeatureVectorMapping);
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("BIAS")), is(true));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("a")), is(true));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("is")), is(true));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("test")), is(true));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("this")), is(true));

        newSampleFeatureVectorMapping = fe.removeHapaxLegomena(sampleFeatureVectorMapping, 3);
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("BIAS")), is(true));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("a")), is(false));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("is")), is(false));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("test")), is(true));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("this")), is(false));

        newSampleFeatureVectorMapping = fe.topFrequentFeatures(sampleFeatureVectorMapping, 2);
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("BIAS")), is(true));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("a")), is(false));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("is")), is(false));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("test")), is(true));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("this")), is(false));

        newSampleFeatureVectorMapping = fe.topFrequentFeatures(sampleFeatureVectorMapping, 3);
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("BIAS")), is(true));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("a")), is(false));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("is")), is(true));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("test")), is(true));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("this")), is(true));

        newSampleFeatureVectorMapping = fe.onlyFrequentFeatures(sampleFeatureVectorMapping, 2);
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("BIAS")), is(true));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("a")), is(false));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("is")), is(true));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("test")), is(true));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("this")), is(true));

        newSampleFeatureVectorMapping = fe.onlyFrequentFeatures(sampleFeatureVectorMapping, 3);
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("BIAS")), is(true));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("a")), is(false));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("is")), is(false));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("test")), is(true));
        assertThat(newSampleFeatureVectorMapping.get(a).getFeatureValues().keySet().contains(lexicon.get("this")), is(false));


    }

}
