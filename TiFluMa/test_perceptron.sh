# mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-perceptron -eval /home/artist/songs_test.txt -wcm -wr -l 1 -e 100 -d 300 -ss -cw -cpos -tf -ew -rh -mst 0" > perceptron_full_log.txt
# mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-perceptron -eval /home/artist/songs_test.txt -wcm -wr -l 1 -e 100 -d 300 -ss -cw -cpos -tf -ew -rh -rao -mst 0" > perceptron_full_red_log.txt
# mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-perceptron -eval /home/artist/songs_test.txt -wcm -wr -l 1 -e 100 -d 300 -ss -cw -cpos -tf -ew -rh -rao -mst 140" > perceptron_red_log.txt
mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-perceptron -eval /home/artist/songs_test.txt -wcm -wr -l 1 -e 100 -d 300 -cn -rao -mst 0" >perceptron_full_red_basic_log.txt
mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-perceptron -eval /home/artist/songs_test.txt -wcm -wr -l 1 -e 100 -d 300 -cn -rao -mst 140" >perceptron_red_basic_log.txt
