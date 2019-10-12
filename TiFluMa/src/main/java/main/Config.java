package main;

/**
 * Hold some static values that can be changed centralised.
 */
public class Config {

    // User Intent
    public static UserIntent userIntent = UserIntent.PERCEPTRON;

    // Files
    public static String trainFile = "/home/artist/songs_train.txt";
    public static String evalFile = "/home/artist/songs_dev.txt";
    public static String testFile = "/home/artist/songs_test.txt";
    public static String featureExportFilePath = "";

    // Setup
    public static boolean writeConfusionmatrixToFile = false;
    public static boolean writeResultsToFile = false;
    public static String fileIdentifier = null;
    public static String folderForKerasEval = "";

    // Hyperparameters
    public static double LEARNING_RATE = 1; // starting update step of the weights, changes throughout learning
    public static double LEARNING_RATE_ADAPTATION = 1; // in the current implementation this is overwritten dynamically,
    // depending on the epochs
    public static int EPOCHS = 30;
    public static int BATCH_SIZE = Integer.MAX_VALUE;
    public static int WORD_VEC_DIMENSIONS = 300;
    public static int MIN_WORD_FREQUENCY = 10;
    public static int WINDOW_SIZE = 8;

    // Feature Selection
    public static boolean useSimpleStylometricFeatures = false; // basic features like "how many lines" or "average
    // length of lines"
    public static boolean useRhymeFeature = false; // use simplistic stylometric normalised rhyme count
    public static boolean useNounCountVectors = false; // count vector containing only nouns
    public static boolean useWordCountVectors = false; // count vector containing all words
    public static boolean usePOSCountVectors = false; // count vector containing all POS tags
    public static boolean tfWeightingForCountVectors = false; // use TF weighting for all count vectors
    public static boolean useNounEmbeddings = false; // use word embeddings of only nouns
    public static boolean useWordEmbeddings = false; // use word embeddings of all words

    // Data Pruning
    public static boolean useRelevantArtistsOnly = false; // remove all artists that have more than x songs in the train
    // set
    public static int artistsHaveToHaveMoreSongsThan = 140; // 140 gives us the 40 most frequent classes, that's a nice
    // value
}
