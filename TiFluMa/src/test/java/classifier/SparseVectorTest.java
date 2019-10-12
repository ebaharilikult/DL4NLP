package classifier;

import main.Config;
import org.junit.Test;

import static org.hamcrest.core.Is.is;
import static org.junit.Assert.assertThat;

public class SparseVectorTest {

    @Test
    public void testVectorMath() {

        // Create vector1 = (1.0, 2.0, 3.0)
        SparseVector vector1 = new WeightVector();
        vector1.addFeature(0);
        vector1.setFeatureValue(0, 1.0);
        vector1.addFeature(1);
        vector1.setFeatureValue(1, 2.0);
        vector1.addFeature(2);
        vector1.setFeatureValue(2, 3.0);

        // Create vector2 = (-1.0, 0.0, 1.0)
        SparseVector vector2 = new FeatureVector();
        vector2.addFeature(0);
        vector2.setFeatureValue(0, -1.0);
        vector2.addFeature(1);
        vector2.setFeatureValue(1, 0.0);
        vector2.addFeature(2);
        vector2.setFeatureValue(2, 1.0);

        // multiplyVector: vector3 = (-1.0, 0.0, 3.0)
        SparseVector vector3 = vector1.multiplyVector(vector2);
        assertThat(vector3.getFeatureValue(0), is(-1.0));
        assertThat(vector3.getFeatureValue(1), is(0.0));
        assertThat(vector3.getFeatureValue(2), is(3.0));

        // multiplyScalar: vector4 = (1.5, 3.0, 4.5)
        SparseVector vector4 = vector1.multiplyScalar(1.5);
        assertThat(vector4.getFeatureValue(0), is(1.5));
        assertThat(vector4.getFeatureValue(1), is(3.0));
        assertThat(vector4.getFeatureValue(2), is(4.5));

        // addVector: vector5 = (0.0, 2.0, 4.0)
        SparseVector vector5 = vector1.addVector(vector2);
        assertThat(vector5.getFeatureValue(0), is(0.0));
        assertThat(vector5.getFeatureValue(1), is(2.0));
        assertThat(vector5.getFeatureValue(2), is(4.0));

        // scalarProduct: scalar = 2.0
        double scalar = vector1.scalarProduct(vector2);
        assertThat(scalar, is(2.0));

        Config.LEARNING_RATE = 1;
        Config.LEARNING_RATE_ADAPTATION = 0;
        // update: vector1 = (0.0, 1.0, 2.0)
        ((WeightVector) vector1).update((FeatureVectorInterface) vector2, Sign.POS);
        assertThat(vector1.getFeatureValue(0), is(0.0));
        assertThat(vector1.getFeatureValue(1), is(2.0));
        assertThat(vector1.getFeatureValue(2), is(4.0));
    }

}
