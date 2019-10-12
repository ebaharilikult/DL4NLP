import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

path_to_train_file = "src/main/resources/songs_dev.txt"
path_to_model = "d2v.model"

def tokenize(doc):
	tokens = word_tokenize(doc.lower())
	return tokens

def train_model():
	with open(path_to_train_file) as f:
		lines = f.readlines()
		lines = [line.strip().split("\t") for line in lines]
		documents = [TaggedDocument(tags=[line[0]+" | "+line[1]], words=tokenize(line[2])) for line in lines]
		model = Doc2Vec(documents, vector_size=150, dm=1, window=10, min_count=1, workers=20)
		model.train(documents, total_examples=len(documents), epochs=100)
		model.save(path_to_model)

def load_model():
	if not os.path.exists(path_to_model):
		train_model()
	model = Doc2Vec.load(path_to_model)
	return model

model = load_model()

def get_known_vector(author, title):
	return model.docvecs[author+" | "+title]

def get_unknown_vector(text):
	return model.infer_vector(tokenize(text))
