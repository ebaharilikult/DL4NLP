import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


def plot_results_per_epoch(path_to_csv):
    with open(path_to_csv) as res:
        results = res.read()

    epochs = list()
    macro_f_score = list()
    micro_f_score = list()

    labels = True

    for line in results.split("\n"):
        if labels or line == "":
            labels = False
            epochs.append(0)
            macro_f_score.append(0.0)
            micro_f_score.append(0.0)
        else:
            epochs.append(int(line.split()[0]) + 1)
            macro_f_score.append(float(line.split()[3]))
            micro_f_score.append(float(line.split()[4]))

    plt.figure(figsize=(27, 9))
    plt.xlabel('Epoch')
    plt.ylabel('F-Score')

    plt.plot(epochs, macro_f_score, label="Macro f-score")
    plt.plot(epochs, micro_f_score, label="Micro f-score")
    plt.title("Learning Progress")
    plt.draw()
    plt.legend()
    plt.show()
    plt.savefig("208_epochs_wide.png", dpi=300)


def plot_confusion_matrix(path_to_csv):
    with open(path_to_csv) as res:
        matrix = res.read()

    rows = list()
    cols = list()
    labels = list()
    first_line = True

    for line in matrix.split("\n"):
        if first_line or line == "":
            # first line is labels
            first_line = False
            for label in line.split("\t")[1::]:
                if label != "":
                    if len(label) > 8:
                        labels.append(label[:8:])
                    else:
                        labels.append(label)
        else:
            for el in line.split("\t")[1::]:
                if el != "":
                    cols.append(float(el))
        if cols != []:
            rows.append(cols)
            cols = list()
    df_cm = pd.DataFrame(rows, index=[i for i in labels], columns=[i for i in labels])
    plt.figure()
    sn.heatmap(df_cm, annot=False, cmap='BuGn', cbar=True, square=True, xticklabels=True, yticklabels=True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_confusion_matrix("01_140_samples_per_class_30_epochs_batch_learning_all_features.csv")
    # plot_results_per_epoch("epochs_208.csv")
