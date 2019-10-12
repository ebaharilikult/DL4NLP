# mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-doc2vec -eval /home/artist/songs_test.txt -wcm -wr -l 0.025 -e 100 -d 300 -f 10 -w 8 -mst 0" > doc2vec_full_log.txt
mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-doc2vec -eval /home/artist/songs_test.txt -wcm -wr -l 0.025 -e 100 -d 300 -f 10 -w 8 -rao -mst 0" >doc2vec_full_red_log.txt
mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-doc2vec -eval /home/artist/songs_test.txt -wcm -wr -l 0.025 -e 100 -d 300 -f 10 -w 8 -rao -mst 140" >doc2vec_red_log.txt
# mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-doc2vec -eval /home/artist/songs_test.txt -wcm -wr -l 0.025 -e 100 -d 300 -f 10 -w 8 -rao -mst 140 -b 64" > doc2vec_red_b64_log.txt
# mvn -e exec:java -Dexec.mainClass="main.Main" -Dexec.args="-doc2vec -eval /home/artist/songs_test.txt -wcm -wr -l 0.025 -e 100 -d 300 -f 10 -w 8 -rao -mst 0 -b 64" > doc2vec_full_red_b64_log.txt
