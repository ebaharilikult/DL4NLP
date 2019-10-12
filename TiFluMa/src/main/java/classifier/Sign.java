package classifier;

/**
 * Mathematical sign: positive (1) or negative (-1).
 */
public enum Sign {
    POS(1), NEG(-1);

    private final int value;

    Sign(final int newValue) {
        value = newValue;
    }

    public int getValue() {
        return value;
    }
}
