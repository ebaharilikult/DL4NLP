package classifier;

/**
 * A sample consists of a (gold) label, a title, and a text.
 */
public class Sample {
    private String label;
    private String title;
    private String text;

    public Sample() {
    }

    public Sample(String label, String title, String text) {
        this.label = label;
        this.title = title;
        this.text = text;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

}
