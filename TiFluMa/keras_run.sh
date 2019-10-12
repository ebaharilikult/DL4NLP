#!/bin/bash
#
# Trains the Deep NN+ model and finally applies the model to the test data.
# Checks if the required features have been exported and transformed and does it if not.

dataDir="features_mst_140"
pythonDir="src/main/python/deep"

export_features_if_necessary() {
    dir=$1
    if [ ! -d $dir ]; then
        echo "Exporting features..."
        sh exportFeatures.sh
    fi
}

#######################################
# Activates the virtual python environment.
# If no environment exists, it will create a new one and installs all required dependencies
# Arguments:
#   directory path to the virtual environment
#######################################
setup_and_activate_python() {
    dir=$1
    if [ ! -d "$dir/venv" ]; then
        echo "Create virtual environment: $dir"
        python3 -m venv "$dir/venv"
        echo "Activate virtual environment..."
        source "$dir/venv/bin/activate"
        echo "Install dependencies... "
        pip install -r "$dir/requirements.txt"
    else
        echo "Activate virtual environment..."
        source "$dir/venv/bin/activate"
    fi
}

train_and_evaluate_deepNN() {
    python $pythonDir/train_concat.py $(readlink -fn $dataDir) $(readlink -fn ".") "deepNNPlus_140"
}

main() {
    absoluteDataPath= readlink -fn $dataDir
    setup_and_activate_python $pythonDir

    export_features_if_necessary $dataDir
    train_and_evaluate_deepNN
}

main "$@"
