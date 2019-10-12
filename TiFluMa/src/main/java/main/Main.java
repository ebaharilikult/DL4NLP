package main;

import classifier.Classifier;
import classifier.ClassifierInterface;
import classifier.Doc2Vec;
import evaluationMethods.Metrics;
import org.apache.commons.cli.*;
import org.apache.uima.resource.ResourceInitializationException;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;

public class Main {

    public static void main(String[] args) throws IOException, InterruptedException, ResourceInitializationException {
        System.setProperty(org.slf4j.impl.SimpleLogger.DEFAULT_LOG_LEVEL_KEY, "INFO");
        System.setProperty("org.slf4j.simpleLogger.logFile", "System.out");

        // options are read and set through command line parameters
        setCommandOptions(args);

        Instant start = Instant.now();

        // creating the classifier object
        if (Config.userIntent == UserIntent.FEATURE_EXPORT) {
            FeatureTransform.transformFeatures();
        } else if (Config.userIntent == UserIntent.EVALUATE_KERAS) {
            evaluateKeras();
        } else {

            ClassifierInterface cl = null;
            switch (Config.userIntent) {
                case PERCEPTRON:
                    cl = new Classifier(new File(Config.trainFile).getAbsolutePath(),
                            new File(Config.evalFile).getAbsolutePath());
                    break;
                case DOC2VEC:
                    cl = new Doc2Vec(new File(Config.trainFile).getAbsolutePath(),
                            new File(Config.evalFile).getAbsolutePath());
                    break;
            }

            // doing the thing
            cl.initialiseClassifier();
            cl.trainClassifier(true);
            cl.runClassifier();

            // creating evaluation object and evaluating
            Metrics mt = new Metrics(new File(Config.evalFile).getAbsolutePath(), cl.getClassifierResults());

            mt.readGoldFile();
            mt.evaluateSamples();

            System.out.println("Precision:              " + mt.retrieveMacroPrecision());
            System.out.println("Recall:                 " + mt.retrieveMacroRecall());
            System.out.println("Macro Averaged F-Score: " + mt.retrieveMacroFScore());
            System.out.println("Micro Averaged F-Score: " + mt.retrieveMicroFScore());


        }

        Instant finish = Instant.now();

        // how many seconds the code took to run. (This number makes us sad)
        long timeElapsed = Duration.between(start, finish).getSeconds();
        System.out.println("Duration: " + timeElapsed);
    }

    /**
     * Reading the parameters from the command line and setting them in the Config
     *
     * @param args
     */
    public static void setCommandOptions(String[] args) {
        Options options = new Options();

        // User Intent
        options.addOption(new Option("perceptron", "perceptron", false, "perceptron"));
        options.addOption(new Option("doc2vec", "doc2vec", false, "doc2vec"));
        options.addOption(new Option("feature_export", "feature_export", false, "feature export"));
        options.addOption(new Option("evaluate_keras", "evaluate_keras", false, "evaluate_keras"));

        // Files
        options.addOption(new Option("train", "train_file", true, "training file"));
        options.addOption(new Option("eval", "eval_file", true, "evaluation file"));
        options.addOption(new Option("test", "test_file", true, "test file"));
        options.addOption(new Option("export", "export_dir", true, "export directory"));

        // Setup
        options.addOption(new Option("wcm", "confusion_matrix", false, "write confusion matrix to file"));
        options.addOption(new Option("wr", "results", false, "write results to file"));

        // Hyperparameters
        options.addOption(new Option("l", "learning_rate", true, "learning rate"));
        options.addOption(new Option("a", "learning_rate_adaption", true, "learning rate adaption"));
        options.addOption(new Option("e", "epochs", true, "training epochs"));
        options.addOption(new Option("b", "batch_size", true, "batch size"));
        options.addOption(new Option("d", "dim", true, "word embedding dimensionality"));
        options.addOption(new Option("f", "min_freq", true, "minimum word frequency"));
        options.addOption(new Option("w", "window", true, "window size"));

        // Feature selection
        options.addOption(new Option("ss", "simple_stylometric", false, "use simple stylometric features"));
        options.addOption(new Option("cn", "noun_count_vectors", false, "use noun count vectors"));
        options.addOption(new Option("cw", "word_count_vectors", false, "use word count vectors"));
        options.addOption(new Option("cpos", "pos_count_vectors", false, "use POS count vectors"));
        options.addOption(new Option("tf", "tf_weighting", false, "TF weighting for count vectors"));
        options.addOption(new Option("en", "noun_embeddings", false, "use noun embeddings"));
        options.addOption(new Option("ew", "word_embeddings", false, "use word embeddings"));
        options.addOption(new Option("rh", "rhyme", false, "rhyme features"));

        // Data Pruning
        options.addOption(new Option("rao", "relevant_artists", false, "use relevant artists only"));
        options.addOption(new Option("mst", "more_songs_than", true, "artists have to have more songs than"));

        for (Option option : options.getOptions()) {
            option.setRequired(false);
        }

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd;
        try {
            cmd = parser.parse(options, args);

            if (!(cmd.hasOption("perceptron") ^ cmd.hasOption("doc2vec") ^ cmd.hasOption("feature_export") ^ cmd.hasOption("evaluate_keras"))) {
                throw new IllegalArgumentException("You must specify exactly one user intent!");
            }
            if (cmd.hasOption("perceptron"))
                Config.userIntent = UserIntent.PERCEPTRON;
            else if (cmd.hasOption("doc2vec"))
                Config.userIntent = UserIntent.DOC2VEC;
            else if (cmd.hasOption("feature_export"))
                Config.userIntent = UserIntent.FEATURE_EXPORT;
            else if (cmd.hasOption("evaluate_keras"))
                Config.userIntent = UserIntent.EVALUATE_KERAS;

            if (cmd.hasOption("confusion_matrix"))
                Config.writeConfusionmatrixToFile = true;
            if (cmd.hasOption("results"))
                Config.writeResultsToFile = true;

            if (cmd.hasOption("train_file"))
                Config.trainFile = cmd.getOptionValue("train_file");
            if (cmd.hasOption("eval_file"))
                Config.evalFile = cmd.getOptionValue("eval_file");
            if (cmd.hasOption("test_file"))
                Config.evalFile = cmd.getOptionValue("test_file");
            if (cmd.hasOption("export_dir"))
                Config.featureExportFilePath = cmd.getOptionValue("export_dir");

            if (cmd.hasOption("learning_rate"))
                Config.LEARNING_RATE = Double.valueOf(cmd.getOptionValue("learning_rate"));
            if (cmd.hasOption("learning_rate_adaption"))
                Config.LEARNING_RATE_ADAPTATION = Double.valueOf(cmd.getOptionValue("learning_rate_adaption"));
            if (cmd.hasOption("epochs"))
                Config.EPOCHS = Integer.valueOf(cmd.getOptionValue("epochs"));
            if (cmd.hasOption("batch_size"))
                Config.BATCH_SIZE = Integer.valueOf(cmd.getOptionValue("batch_size"));
            if (cmd.hasOption("dim"))
                Config.WORD_VEC_DIMENSIONS = Integer.valueOf(cmd.getOptionValue("dim"));
            if (cmd.hasOption("min_freq"))
                Config.MIN_WORD_FREQUENCY = Integer.valueOf(cmd.getOptionValue("min_freq"));
            if (cmd.hasOption("window"))
                Config.WINDOW_SIZE = Integer.valueOf(cmd.getOptionValue("window"));

            if (cmd.hasOption("simple_stylometric"))
                Config.useSimpleStylometricFeatures = true;
            if (cmd.hasOption("noun_count_vectors"))
                Config.useNounCountVectors = true;
            if (cmd.hasOption("word_count_vectors"))
                Config.useWordCountVectors = true;
            if (cmd.hasOption("pos_count_vectors"))
                Config.usePOSCountVectors = true;
            if (cmd.hasOption("tf"))
                Config.tfWeightingForCountVectors = true;
            if (cmd.hasOption("noun_embeddings"))
                Config.useNounEmbeddings = true;
            if (cmd.hasOption("word_embeddings"))
                Config.useWordEmbeddings = true;
            if (cmd.hasOption("rh"))
                Config.useRhymeFeature = true;

            if (cmd.hasOption("relevant_artists"))
                Config.useRelevantArtistsOnly = true;
            if (cmd.hasOption("more_songs_than"))
                Config.artistsHaveToHaveMoreSongsThan = Integer.valueOf(cmd.getOptionValue("more_songs_than"));

        } catch (org.apache.commons.cli.ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("Main", options);
            System.exit(1);
        }

    }

    /**
     * Build an ArrayList<String> from a file and evaluate said file for our Keras model
     */
    private static void evaluateKeras() throws IOException {
        // filenames are hardcoded here
        String pathToGold = "gold.txt";
        String pathToResults = "results.txt";
        ArrayList<String> kerasResults = new ArrayList<>();
        // folder names are part of config
        Files.lines(Paths.get(Config.folderForKerasEval + pathToResults)).forEach(kerasResults::add);
        Metrics mt = new Metrics(Config.folderForKerasEval + pathToGold, kerasResults);
        Config.writeConfusionmatrixToFile = true;
        Config.writeResultsToFile = true;
        Config.useRelevantArtistsOnly = false;
        mt.readGoldFile();
        mt.evaluateSamples();
    }

}
