package main;

import java.io.File;
import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import classifier.Classifier;

public class FeatureTransform {

	static Logger logger = LoggerFactory.getLogger(FeatureTransform.class);

	public static void transformFeatures() throws IOException {
		logger.info("main >");

		Classifier cl = new Classifier(new File(Config.trainFile).getAbsolutePath(),
				new File(Config.evalFile).getAbsolutePath(), new File(Config.testFile).getAbsolutePath());

		System.out.println("useNounCountVectors: " + Config.useNounCountVectors);
		System.out.println("useNounEmbeddings: " + Config.useNounEmbeddings);
		System.out.println("usePOSCountVectors: " + Config.usePOSCountVectors);
		System.out.println("useRelevantArtistsOnly: " + Config.useRelevantArtistsOnly);
		System.out.println("useRhymeFeature: " + Config.useRhymeFeature);
		System.out.println("useSimpleStylometricFeatures: " + Config.useSimpleStylometricFeatures);
		System.out.println("useWordCountVectors: " + Config.useWordCountVectors);
		System.out.println("useWordEmbeddings: " + Config.useWordEmbeddings);
		System.out.println("featureExportFilePath: " + Config.featureExportFilePath);

		// train needs to be executed first so that we have a correct feature count
		cl.extractTrainFeaturesToFile(Config.featureExportFilePath + "/train_features.zip");
		cl.extractEvaluationFeaturesToFile(Config.featureExportFilePath + "/eval_features.zip");
		cl.extractTestFeaturesToFile(Config.featureExportFilePath + "/test_features.zip");

		logger.info("info <");
	}
}
