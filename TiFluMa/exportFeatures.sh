#!/bin/bash
#
# Calls the feature exporter and the featurer transformer for each feature.
# The transformed features will be stored inside the specified base directory

baseDir="features_mst_140"
pythonDir="src/main/python/deep"

create_dir_if_not_exists() {
  dir=$1
  if [ ! -d $dir ]; then
    echo "Create directory: $dir"
    mkdir $dir
  fi
}

export_and_transform_stylometric_features() {
  dir="$baseDir/stylometric"
  create_dir_if_not_exists $dir
  mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-feature_export -rao -mst 140 -tf -ss -export_dir $dir"
  python $pythonDir/featureTransformer.py $dir
}
export_and_transform_rhyme_features() {
  dir="$baseDir/rhyme"
  create_dir_if_not_exists $dir
  mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-feature_export -rao -mst 140 -tf -rh -export_dir $dir"
  python $pythonDir/featureTransformer.py $dir
}

export_cnd_transform_noun_count_vector_features() {
  dir="$baseDir/noun_count_vectors"
  create_dir_if_not_exists $dir
  mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-feature_export -rao -mst 140 -tf -cn -export_dir $dir"
  python $pythonDir/featureTransformer.py $dir
}

export_and_transform_word_count_vector_features() {
  dir="$baseDir/word_count_vectors"
  create_dir_if_not_exists $dir
  mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-feature_export -rao -mst 140 -tf -cw -export_dir $dir"
  python $pythonDir/featureTransformer.py $dir
}

export_and_transform_pos_count_vectorsFeatures() {
  dir="$baseDir/pos_count_vectors"
  create_dir_if_not_exists $dir
  mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-feature_export -rao -mst 140 -tf -cpos -export_dir $dir"
  python $pythonDir/featureTransformer.py $dir
}

export_and_transform_word_embeddings_features() {
  dir="$baseDir/word_embeddings"
  create_dir_if_not_exists $dir
  mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-feature_export -rao -mst 140 -tf -ew -export_dir $dir"
  python $pythonDir/featureTransformer.py $dir
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

main() {
  setup_and_activate_python $pythonDir
  which python

  # Create export dir
  create_dir_if_not_exists $baseDir

  # export and transform features
  export_and_transform_stylometric_features
  export_and_transform_rhyme_features
  export_and_transform_word_count_vector_features
  export_cnd_transform_noun_count_vector_features
  export_and_transform_pos_count_vectorsFeatures
  export_and_transform_word_embeddings_features
}

main "$@"
