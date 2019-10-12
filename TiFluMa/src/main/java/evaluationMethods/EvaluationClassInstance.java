package evaluationMethods;

public class EvaluationClassInstance {

    private int truePositives;
    private int falsePositives;
    private int trueNegatives;
    private int falseNegatives;

    public EvaluationClassInstance() {
    }

    public int getTruePositives() {
        return truePositives;
    }

    public void setTruePositives(int truePositives) {
        this.truePositives = truePositives;
    }

    public int getFalsePositives() {
        return falsePositives;
    }

    public void setFalsePositives(int falsePositives) {
        this.falsePositives = falsePositives;
    }

    public int getTrueNegatives() {
        return trueNegatives;
    }

    public void setTrueNegatives(int trueNegatives) {
        this.trueNegatives = trueNegatives;
    }

    public int getFalseNegatives() {
        return falseNegatives;
    }

    public void setFalseNegatives(int falseNegatives) {
        this.falseNegatives = falseNegatives;
    }

    /**
     * increases the false negatives of this class by one
     */
    public void increaseFalseNegatives() {
        ++this.falseNegatives;
    }

    /**
     * increases the true negatives of this class by one
     */
    public void increaseTrueNegatives() {
        ++this.trueNegatives;
    }

    /**
     * increases the false positives of this class by one
     */
    public void increaseFalsePositives() {
        ++this.falsePositives;
    }

    /**
     * increases the true positives of this class by one
     */
    public void increaseTruePositives() {
        ++this.truePositives;
    }

    /**
     * Calculates and returns the precision.
     *
     * @return calculated precision
     */
    public double precision() {
        if (this.truePositives + this.falsePositives == 0) {
            return 0.0;
        }
        return 1.0 * this.truePositives / (this.truePositives + this.falsePositives);
    }

    /**
     * Calculates and returns the recall.
     *
     * @return calculated recall
     */
    public double recall() {
        if (this.truePositives + this.falseNegatives == 0) {
            return 0.0;
        }
        return 1.0 * this.truePositives / (this.truePositives + this.falseNegatives);
    }

    /**
     * Calculates and returns the f-score.
     *
     * @return calculated f-score
     */
    public double fScore() {
        if (this.precision() + this.recall() == 0) {
            return 0.0;
        }
        return 2.0 * this.precision() * this.recall() / (this.precision() + this.recall());
    }

    /**
     * Calculates and returns the accuracy.
     *
     * @return calculated accuracy
     */
    public double accuracy() {
        if (this.truePositives + this.trueNegatives + this.falsePositives + this.falseNegatives == 0) {
            return 0.0;
        }
        return 1.0 * (this.truePositives + this.trueNegatives) / (this.truePositives + this.trueNegatives
                + this.falsePositives + this.falseNegatives);
    }

    @Override
    public String toString() {
        return "EvaluationClassInstance{" +
                "truePositives=" + truePositives +
                ", falsePositives=" + falsePositives +
                ", trueNegatives=" + trueNegatives +
                ", falseNegatives=" + falseNegatives +
                '}';
    }
}
