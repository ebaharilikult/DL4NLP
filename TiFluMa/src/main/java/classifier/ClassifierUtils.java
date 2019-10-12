package classifier;

import main.Config;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.HashSet;
import java.util.List;

public class ClassifierUtils {

    public static HashSet<String> extractRelevantArtists(String trainingFilePath, String evaluationFilePath) throws IOException {
        Classifier cl = new Classifier();
        List<Sample> train_samples = cl.readSamplesFromFile(trainingFilePath);
        List<Sample> eval_samples = cl.readSamplesFromFile(evaluationFilePath);
        HashSet<String> eval_classes = new HashSet<>();
        HashSet<String> train_classes = new HashSet<>();
        LinkedHashMap<String, Integer> songsPerArtist = new LinkedHashMap<>();
        HashSet<String> relevantArtists = new HashSet<>();
        for (Sample sample : train_samples) {
            train_classes.add(sample.getLabel());
            int prevCount = songsPerArtist.getOrDefault(sample.getLabel(), 0);
            songsPerArtist.put(sample.getLabel(), prevCount + 1);
        }
        for (Sample sample : eval_samples) {
            eval_classes.add(sample.getLabel());
        }
        HashSet<String> intersected_classes = new HashSet<>(train_classes);
        intersected_classes.retainAll(eval_classes);
        for (String artist : intersected_classes) {
            if (songsPerArtist.get(artist) > Config.artistsHaveToHaveMoreSongsThan) {
                relevantArtists.add(artist);
            }
        }
        return relevantArtists;
    }

}
