import gensim
import re
import sys
from string import ascii_lowercase

whitespaces = re.compile("\s+")

def preprocess_song_text(text):
	"""Equivalent to FeatureExtractor.java's preprocessSongText().
		In: text
		Out: text
	"""
	replacements = [
		("[", " "),
		("]", " "),
		("NEWLINE", " "),
		("(", " "),
		(")", " "),
		("Chorus", " "),
		("CHORUS", " "),
		("Verse", " "),
		("VERSE", " "),
		(".", " "),
		(":", " "),
		(",", " "),
		(";", " "),
		("*", " "),
		("-", " "),
		("!", " "),
		("?", " "),
		("\"", " "),
		("#", " "),
		("/", " "),
		("!", " "),
		("'", " '")
	]
	for replacement in replacements:
		text = text.replace(replacement[0], replacement[1])
	return text

def postprocess_song_text(tokens):
	"""Equivalent to FeatureExtractor.java's postprocessSongText().
		Makes substitutions of the form yyyaaaaayyyy" ->  "yyyaaayyy".
		In: list of tokens
		Out: list of tokens
	"""
	for i, token in enumerate(tokens):
		for c in ascii_lowercase:
			tokens[i] = re.sub(4*c+"+", 3*c, token)
	return tokens

def tokenise(text):
	"""Equivalent to FeatureExtractor.java's tokenise().
		In: text
		Out: list of tokens
	"""
	text = preprocess_song_text(text)
	text = whitespaces.sub(text, " ").strip()
	tokens = text.split(" ")
	tokens = [token[0:1]+(token[1:].lower() if len(token) > 1 else "") for token in tokens]
	tokens = postprocess_song_text(tokens)
	return tokens

def read_input(filename):
	"""Read a songs_*.txt file.
		In: filename
		Out: list of documents, a document is a list of tokens
	"""
	documents = []
	with open(filename) as f:
		for text in f:
			text = text.strip().split("\t")[2]
			tokens = tokenise(text)
			documents.append(tokens)
	return documents

def build_model(corpus_file, result_file):
	"""Trains word2vec vectors on a given songs_*.txt file.
		In: path to corpus file, path to save location
	"""
	documents = read_input(corpus_file)
	model = gensim.models.Word2Vec(documents, size=300, window=11, min_count=10, workers=40)
	model.train(documents, total_examples=len(documents), epochs=15)
	model.wv.save_word2vec_format(result_file, binary=False)

if __name__ == "__main__":
	args = sys.argv
	if len(args) == 3:
		corpus_file = args[1]
		result_file = args[2]
		build_model(corpus_file, result_file)
