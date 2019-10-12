mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-perceptron -e 5" > example_log.txt
mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-doc2vec -e 5" >> example_log.txt

