package evaluationMethods;

import org.junit.Test;

import static org.hamcrest.core.Is.is;
import static org.hamcrest.number.IsCloseTo.closeTo;
import static org.junit.Assert.assertThat;

public class EvaluationClassInstanceTest {

    EvaluationClassInstance eci;

    @Test
    public void testPerformanceMeasures() {
        this.eci = new EvaluationClassInstance();

        // test for null values
        assertThat(this.eci.accuracy(), is(0.0));
        assertThat(this.eci.fScore(), is(0.0));
        assertThat(this.eci.recall(), is(0.0));

        // test for simple accuracy
        increaseFalseNegatives(1);
        increaseTrueNegatives(1);

        assertThat(this.eci.accuracy(), is(0.5));
        assertThat(this.eci.recall(), is(0.0));
        assertThat(this.eci.fScore(), is(0.0));

        // test complex example
        increaseFalseNegatives(3);
        increaseFalsePositives(27);
        increaseTruePositives(8);
        increaseTrueNegatives(4);

        assertThat(this.eci.getFalseNegatives(), is(4));
        assertThat(this.eci.getFalsePositives(), is(27));
        assertThat(this.eci.getTruePositives(), is(8));
        assertThat(this.eci.getTrueNegatives(), is(5));
        assertThat(this.eci.accuracy(), closeTo(0.29, 0.1));
        assertThat(this.eci.fScore(), closeTo(0.34, 0.1));
    }

    /**
     * increases the true negatives by the given amount
     *
     * @param amount
     *            defines how much the true negatives should be increased
     */
    private void increaseTrueNegatives(int amount) {
        for (int i = 0; i < amount; i++) {
            this.eci.increaseTrueNegatives();
        }
    }

    /**
     * increases the true positives by the given amount
     *
     * @param amount
     *            defines how much the true positives should be increased
     */
    private void increaseTruePositives(int amount) {
        for (int i = 0; i < amount; i++) {
            this.eci.increaseTruePositives();
        }
    }

    /**
     * increases the false negatives by the given amount
     *
     * @param amount
     *            defines how much the false negatives should be increased
     */
    private void increaseFalseNegatives(int amount) {
        for (int i = 0; i < amount; i++) {
            this.eci.increaseFalseNegatives();
        }
    }

    /**
     * increases the false positives by the given amount
     *
     * @param amount
     *            defines how much the false positives should be increased
     */
    private void increaseFalsePositives(int amount) {
        for (int i = 0; i < amount; i++) {
            this.eci.increaseFalsePositives();
        }
    }
}
