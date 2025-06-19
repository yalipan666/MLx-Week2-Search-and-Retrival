#
#
#
import collections
import pickle


#
#
#
with open('./corpus/text8.txt') as f: text8: str = f.read()
with open('./corpus/msmarco.txt') as f: msmarco: str = f.read()
#  with open, by default is 'r'
# The : str is a type hint, indicating that text8 is expected to be a string.

#
#
#
def preprocess(text: str) -> list[str]:
  #  can also be written just as def preprocess(text), 
  # but his is less informative, and in modern Python (3.5+), 
  # type hints are encouraged
  text = text.lower()
  text = text.replace('.',  ' <PERIOD> ')
  text = text.replace(',',  ' <COMMA> ')
  text = text.replace('"',  ' <QUOTATION_MARK> ')
  text = text.replace('“',  ' <QUOTATION_MARK> ')
  text = text.replace('”',  ' <QUOTATION_MARK> ')
  text = text.replace(';',  ' <SEMICOLON> ')
  text = text.replace('!',  ' <EXCLAMATION_MARK> ')
  text = text.replace('?',  ' <QUESTION_MARK> ')
  text = text.replace('(',  ' <LEFT_PAREN> ')
  text = text.replace(')',  ' <RIGHT_PAREN> ')
  text = text.replace('--', ' <HYPHENS> ')
  text = text.replace('?',  ' <QUESTION_MARK> ')
  text = text.replace(':',  ' <COLON> ')
  text = text.replace("'",  ' <APOSTROPHE> ')
  text = text.replace("’",  ' <APOSTROPHE> ')
  words = text.split()
  stats = collections.Counter(words)
  words = [word for word in words if stats[word] > 5]
  return words


#
#
#
def create_lookup_tables(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
  word_counts = collections.Counter(words)
  vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
  # sorting from most to least frequency is necessary and expected by modern NLP
  # earier for truncation if needed (e.g., only the top 1000words) or remove rare words
  int_to_vocab = {ii+1: word for ii, word in enumerate(vocab)}
  int_to_vocab[0] = '<PAD>'
  vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
  return vocab_to_int, int_to_vocab
 # both mappings are necessary as one is for encoding and the other one is for decoding 
 # easy to get, rather than construct one from the other everytime needed

#
#
#
corpus: list[str] = preprocess(text8 + msmarco)
words_to_ids, ids_to_words = create_lookup_tables(corpus)
tokeniser = { "words_to_ids": words_to_ids, "ids_to_words": ids_to_words }
tokens: list[int] = [words_to_ids[word] for word in corpus]
with open('./corpus/tokens.txt', 'w', encoding='utf-8') as f: f.write('\n'.join(map(str, tokens)))
# When you want to join the tokens into a single string (e.g., for writing to a file), 
# you need them to be strings, not integers.
with open('./corpus/tokeniser.pkl', 'wb') as f: pickle.dump(tokeniser, f)
# When you use pickle.dump(), you must open the file in binary mode ('wb')
# because pickle writes bytes, not text.

#
#
#
print("VOCAB:", len(words_to_ids))
