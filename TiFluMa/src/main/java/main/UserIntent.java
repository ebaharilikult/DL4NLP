package main;

public enum UserIntent {
    PERCEPTRON("perceptron"), DOC2VEC("doc2vec"), FEATURE_EXPORT("feature_export"), EVALUATE_KERAS("evaluate_keras");

    private final String value;

    UserIntent(final String newValue) {
        value = newValue;
    }

    public String getValue() {
        return value;
    }
}
