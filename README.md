# TiFluMa

TiFluMa represents multiple neural-network based classifier for artist/author classification.

It contains a one-layer perceptron, a two-layer Doc2Vec model and a multi-layer neural network.

The implementation is done as a Java Maven project for perceptron and the Doc2Vec model and in python for the deep neural network.

## Project structure (selection)

```
TiFluMa
├── example_run.sh                                     # run perceptron/doc2vec
├── keras_run.sh                                       # run keras
├── pom.xml                                            # maven project configuration file
└── src
    ├── main
    │   ├── java
    │   │   ├── classifier                             # interfaces/implementations of classifiers and sparse vectors
    │   │   │   ├── ClassifierInterface.java           # abstract classifier interface
    │   │   │   ├── Classifier.java                    # perceptron classifier
    │   │   │   ├── Doc2Vec.java                       # doc2vec classifier
    │   │   │   ├── FeatureExtractor.java
    │   │   │   ├── FeatureVector.java
    │   │   │   ├── Sample.java
    │   │   │   ├── SparseVectorInterface.java         # abstract sparse vector interface
    │   │   │   ├── SparseVector.java
    │   │   │   └── WeightVector.java
    │   │   ├── evaluationMethods                      # precision, recall, f-score implementations
    │   │   │   ├── EvaluationClassInstance.java
    │   │   │   └── Metrics.java
    │   │   └── main
    │   │       ├── Config.java                        # project parameters
    │   │       ├── FeatureTransform.java
    │   │       └── Main.java                          # command line options
    │   ├── python
    │   │   ├── deep                                   # deep nn classifier
    │   │   │   ├── featureTransformer.py
    │   │   │   ├── requirements.txt
    │   │   │   └── train_concat.py
    │   │   └── Plotting                               # data files and plotting scripts
    │   │       └── HarryPlotter.py
    │   └── resources                                  # provided test files
    │       ├── songs_dev_minimized.txt
    │       ├── songs_dev-predicted.txt
    │       └── songs_dev.txt
    └── test                                           # unit tests
        ├── java
        │   ├── classifier
        │   │   ├── ClassifierTest.java
        │   │   ├── Doc2VecTest.java
        │   │   ├── FeatureExtractorTest.java
        │   │   └── SparseVectorTest.java
        │   └── evaluationMethods
        │       ├── EvaluationClassInstanceTest.java
        │       └── MetricsTest.java
        └── resources                                  # own test files
            ├── mock_up_test.txt
            ├── mock_up_train.txt
            └── test_gold_file.txt
```

## Demo evaluation

### Java project

You can run the program via the command line, e.g. as follows:

```
mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-perceptron -e 5"
```

All Config parameters, including training/evaluation files, can be specified within `-Dexec.args`; see the Main class for the corresponding command line option names.

### Keras project

`keras_run.sh` contains calls to setup, train and evaluate the Deep NN+ model in one single script.
For setup, it creates a virtual environment and installs all dependencies if necessary.
Afterwards, it runs the feature exporter and the feature transformer to prepare the features for the use of the Keras model.
Then it calls the neural network to train the Deep NN+ model on the exported features.
Finally, it applies the trained model to the test data.
