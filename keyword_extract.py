# written by kylesim

import sys
import re
import time
import argparse
from collections import Counter, defaultdict
from tokenizer import get_tokenizer

PARAM = None
parser = argparse.ArgumentParser()
parser.add_argument(
	'-a', '--action', type=str,
	default='word', help='word, coword, phrase'
)
parser.add_argument(
	'-t', '--tokenizer', type=str,
	default='mecab', help=''
)
parser.add_argument(
	'-o', '--word_out_pos', type=str,
	default='all', help='output pos for word'
)
parser.add_argument(
	'-s', '--stopwords', type=str,
	default='', help='stopwords file'
)
parser.add_argument(
	'-k', '--top_k', type=int,
	default=100, help=''
)
parser.add_argument(
	'-w', '--window', type=int,
	default=2, help=''
)
parser.add_argument(
	'-c', '--min_cooccurrence', type=int,
	default=1, help=''
)
parser.add_argument(
	'-d', '--dictionary', type=str,
	default='', help='user dictionary file(directory) for komoran(mecab)'
)
parser.add_argument(
	'-i', '--add_index', action='store_true',
	default=False, help=''
)
parser.add_argument(
	'-m', '--merge_pos', action='store_true',
	default=False, help=''
)
PARAM = parser.parse_args()
print(PARAM)

POS_ROOT = ['XR']
POS_NOUN = ['NNB', 'NNBC', 'NNG', 'NNP', 'XSN', 'NR', 'SN', 'SL', 'NNG+XSN', 'NNB+JX']
POS_ADJ = ['VA', 'VA+EC', 'VA+EF', 'VA+EP', 'VA+ETM', 'XSA', 'XSA+ETM']
POS_VERB = ['VV', 'VX', 'XSV', 'VV+EC', 'VV+EF', 'VV+EP', 'VV+ETM', 'VV+ETN', 'VX+EC', 'XSV+ETN', 'XSV+ETM']
POS_POS = ['VCP']
POS_NEG = ['VCN']
POS_EOME = ['EC', 'EF', 'EP', 'ETM', 'ETN']

# WORD
if PARAM.word_out_pos == 'noun':
	WORD_POS_TAGS = POS_NOUN + POS_ROOT
elif PARAM.word_out_pos == 'adj':
	WORD_POS_TAGS = POS_ADJ
else:
	# all = noun + adj
	WORD_POS_TAGS = POS_NOUN + POS_ROOT + POS_ADJ

# WORD, COWORD, PHRASE
OUT_POS_TAGS = POS_NOUN + POS_ROOT + POS_ADJ + POS_VERB + POS_EOME

STOPWORDS = set([])
if PARAM.stopwords:
	def load_stopwords(stopwords_file):
		stop_list = []
		with open(stopwords_file, "r") as f_stop:
			for line in f_stop:
				line = line.strip()
				if line[0] == '#':
					continue
				stop_list.append(line)
		return set(stop_list)
	STOPWORDS = load_stopwords(PARAM.stopwords)


def print_each(items):
	for item in items:
		print(item)


def print_run_time(start_time, print_heading='run'):
	end_time = time.time()
	print("_{}_time: {:.5f}".format(print_heading, end_time - start_time))


def clean_string(string, lower=False):
	string = re.sub(r"[^가-힣A-Za-z0-9().,!?+\-\'\"]", " ", string)
	if lower:
		string = string.lower()
	return string


def get_pos(tokenizer, sentence, add_index=False, debug=False):
	if debug:
		print("\n", sentence)

	try:
		sent_pos = tokenizer.pos(sentence, join=False)
	except:
		sent_pos = tokenizer.pos(clean_string(sentence), join=False)

	if debug:
		print("\n", "* raw_pos")
		print(sent_pos)

	if add_index:
		#(word, pos, index)
		sent_pos = [(word_pos[0], word_pos[1], idx) for idx, word_pos in enumerate(sent_pos) if word_pos[1] in OUT_POS_TAGS]
	else:
		#(word, pos)
		sent_pos = [(word_pos[0], word_pos[1]) for word_pos in sent_pos if word_pos[1] in OUT_POS_TAGS]

	if debug:
		print("\n", "* word_out_pos")
		print(sent_pos)
	return sent_pos


# left & right
def get_cowords_v2(sent_pos, lt_cowords, rt_cowords, window=2):
	sent_len = len(sent_pos)
	print(sent_pos)

	for i in range(sent_len):
		end_i = min(i+window+1, sent_len)
		for j in range(i+1, end_i):
			w_i = "%s/%s" % (sent_pos[i][0], sent_pos[i][1])
			w_j = "%s/%s" % (sent_pos[j][0], sent_pos[j][1])
			#print(w_i, w_j)
			lt_cowords[(w_i, w_j)] += 1
			rt_cowords[(w_j, w_i)] += 1
	return lt_cowords, rt_cowords


def get_cowords(sent_pos, cowords, window=2):
	sent_len = len(sent_pos)
	for i in range(sent_len):
		end_i = min(i+window+1, sent_len)
		for j in range(i+1, end_i):
			### REMOVE POS HERE for SAME WORD, DIFF POS!!! ###
			w_i = "%s/%s" % (sent_pos[i][0], sent_pos[i][1])
			w_j = "%s/%s" % (sent_pos[j][0], sent_pos[j][1])
			#print(w_i, w_j)
			cowords[(w_i, w_j)] += 1
			if i+1 == j:
				# 빠른 속도, 가벼운 무게의 노트북
				cowords[(w_j, w_i)] += 1
				#print(j, i, w_j, w_i)
	return cowords


# rule-based phrases
def get_phrases(sent_pos, window=2):
	words, phrases = [], []
	sent_len = len(sent_pos)
	if sent_len <= 0:
		return phrases
	last_idx = sent_pos[-1][2]
	prev_idx = last_idx
	i = 0
	# (word_0, pos_1, idx_2)
	while i < sent_len:
		word, pos, idx = sent_pos[i]
		diff_idx = idx - prev_idx
		if pos in ['NNG'] and diff_idx <= window:
			prev_idx = idx
			words.append('%s/%s' % (word, pos))
			i += 1
			if i < sent_len:
				word, pos, idx = sent_pos[i]
				diff_idx = idx - prev_idx
				if pos in ['NNG'] and diff_idx <= 1:
					prev_idx = idx
					# NNG + NNG (문서 작업)
					continue
				elif pos in ['VA+ETM', 'VA+EC', 'VA+EF'] and diff_idx <= window:
					words.append('%s/%s' % (word, pos))
					phrases.append(tuple(words))
					# NNG + VA+ETM (가성비 좋은)
				elif pos in ['VV+ETN', 'XSV+ETN'] and diff_idx <= window:
					prev_idx = idx
					words.append('%s/%s' % (word, pos))
					i += 1
					if i < sent_len:
						word, pos, idx = sent_pos[i]
						diff_idx = idx - prev_idx
						if pos in ['VA+ETM', 'VA+EC', 'VA+EF'] and diff_idx <= window:
							words.append('%s/%s' % (word, pos))
							phrases.append(tuple(words))
							# NNG + VV+ETN + VA+ETM (대학생 쓰기 좋은)
							# NNG + NNG + XSV+ETN + VA+ETM (문서작업 하기 좋은)
						else:
							i -= 2
				elif pos in ['XR'] and diff_idx <= window:
					prev_idx = idx
					words.append('%s/%s' % (word, pos))
					i += 1
					if i < sent_len:
						word, pos, idx = sent_pos[i]
						diff_idx = idx - prev_idx
						if pos in ['XSA+ETM'] and diff_idx <= window:
							words.append('%s/%s' % (word, pos))
							phrases.append(tuple(words))
							# NNG + XR + XSA+ETN/XSA+ETM (두께 슬림 한)
						else:
							i -= 2
				else:
					i -= 1
		elif pos in ['VV+EC'] and diff_idx <= window:
			prev_idx = idx
			words.append('%s/%s' % (word, pos))
			i += 1
			if i < sent_len:
				word, pos, idx = sent_pos[i]
				diff_idx = idx - prev_idx
				if pos in ['VV+ETN'] and diff_idx <= window:
					prev_idx = idx
					words.append('%s/%s' % (word, pos))
					i += 1
					if i < sent_len:
						word, pos, idx = sent_pos[i]
						diff_idx = idx - prev_idx
						if pos in ['VA+ETM', 'VA+EC', 'VA+EF'] and diff_idx <= window:
							words.append('%s/%s' % (word, pos))
							phrases.append(tuple(words))
							# VV+EC + VV+ETN + VA+ETM (가지고 다니기 편한)
						else:
							i -= 2
				else:
					i -= 1
		i += 1
		words = []
		prev_idx = last_idx
	return phrases


# NNG+XSN, VA+ETN, VV+ETM
def merge_pos(sent_pos, add_index=False, debug=False):
	merge_sent = []
	merge_next = 0
	for word_pos in sent_pos:
		if word_pos[1] in ['NNG']:
			merge_next = 1
		elif word_pos[1] in ['VA', 'VV', 'VX', 'XSA', 'XSV']:
			merge_next = 2
		elif merge_next > 0:
			check_type = merge_next
			merge_next = 0
			if (check_type == 1 and word_pos[1] in 'XSN') or (check_type == 2 and word_pos[1] in POS_EOME):
				# NNG+XSN, VA+ETN
				word = merge_sent[-1][0] + word_pos[0]
				pos = merge_sent[-1][1] + "+" + word_pos[1]
				if add_index:
					merge_sent[-1] = (word, pos, merge_sent[-1][2])
				else:
					merge_sent[-1] = (word, pos)
				continue
		merge_sent.append(word_pos)
	if debug:
		print("\n", "* merged")
		print(merge_sent)
	return merge_sent


def load_sentences():
	review_count = 0
	sentences = []
	for line in sys.stdin:
		line = line.strip()
		if not line:
			continue
		if line.startswith("__REVIEW__"):
			review_count += 1
			continue
		sentences.append(line)
	print("_review_count: {}".format(review_count))
	print("_sentence_count: {}".format(len(sentences)))
	return sentences


def get_out_pos(sent_pos, debug=False):
	sent_pos = [word_pos for word_pos in sent_pos if word_pos[1] in WORD_POS_TAGS and word_pos[0] not in STOPWORDS]
	if debug:
		print("\n", "* word_out_pos")
		print(sent_pos)
	return sent_pos


def postproc_words(counter, word_out_pos, top_k=10, debug=False):
	if debug:
		print("\n", "_words_stats")
		print(counter.most_common(top_k))
	count = 0
	results = []
	if word_out_pos == 'noun':
		for wp, c in counter.most_common():
			if len(wp[0]) < 2:
				continue
			results.append((wp, c))
			count += 1
			if count >= top_k:
				break
	elif word_out_pos == 'adj':
		for wp, c in counter.most_common():
			if wp[1] not in ['VA+ETM']:
				continue
			results.append((wp, c))
			count += 1
			if count >= top_k:
				break
	else:
		for wp, c in counter.most_common():
			if (wp[1] in ['VA+ETM']) or (wp[1] in POS_NOUN and len(wp[0]) > 1):
				results.append((wp, c))
				count += 1
				if count >= top_k:
					break
	if debug:
		print("\n", "_words_results")
		for i in results:
			print("%s/%s" % (i[0][0], i[0][1]), i[1])
	return results


def postproc_cowords(cowords, top_k, min_cooccurrence, debug=False):
	if debug:
		print("\n", "_cowords_stats")
		print(sorted(cowords.items(), key=lambda x:-x[1])[:PARAM.top_k])

	count = 0
	results = []
	for cw in sorted(cowords.items(), key=lambda x:-x[1]):
		# (left, right): count
		lt_wp = cw[0][0].split("/")
		rt_wp = cw[0][1].split("/")
		if (lt_wp[-1] in POS_NOUN and len(lt_wp[0]) > 1 and lt_wp[0] not in ['만족']) and \
			(rt_wp[-1] in ['VA+ETM', 'XR'] or rt_wp[0] in ['양호', '만족']):
			# (NOUN + ADJ, NOUN + XR)
			if cw[1] < min_cooccurrence:
				break
			results.append(cw)
			count += 1
			if count >= top_k:
				break
	if debug:
		print("\n", "_cowords_results")
		print(results)
	return results


def merge_property(results, add_pos=False, debug=False):
	pv_dic = defaultdict(list)
	pc_dic = defaultdict(int)
	for pv in results:
		if add_pos:
			pc_dic[pv[0][0]] += pv[1] # count
			pv_dic[pv[0][0]].append((pv[0][1], pv[1])) # (r_word/r_pos), count
		else:
			pc_dic[pv[0][0].split("/")[0]] += pv[1] # count
			pv_dic[pv[0][0].split("/")[0]].append((pv[0][1].split("/")[0], pv[1])) # r_word, count

	if debug:
		print("\n", "_property_values")
		for p, c in sorted(pc_dic.items(), key=lambda x:-x[1]):
			print(p, c, pv_dic[p])


def run_words(sentences, tokenizer):
	start_time = time.time()
	counter = Counter()
	for sentence in sentences:
		sent_pos = get_pos(tokenizer, sentence, add_index=False, debug=True)
		if PARAM.merge_pos:
			sent_pos = merge_pos(sent_pos, add_index=False, debug=True)
		sent_pos = get_out_pos(sent_pos, debug=True)
		counter.update(sent_pos)
	results = postproc_words(counter, PARAM.word_out_pos, PARAM.top_k, debug=True)
	print_run_time(start_time, "run")


def run_cowords(sentences, tokenizer):
	start_time = time.time()
	cowords = defaultdict(int)
	for sentence in sentences:
		sent_pos = get_pos(tokenizer, sentence, add_index=False, debug=True)
		if PARAM.merge_pos:
			sent_pos = merge_pos(sent_pos, add_index=False, debug=True)
		get_cowords(sent_pos, cowords, PARAM.window)
	results = postproc_cowords(cowords, PARAM.top_k, PARAM.min_cooccurrence, debug=True)
	merge_property(results, add_pos=False, debug=True)
	#cowords = {key:val for key, val in cowords.items() if val >= PARAM.min_cooccurrence}
	print_run_time(start_time, "run")


def run_phrases(sentences, tokenizer):
	for sentence in sentences:
		sent_pos = get_pos(tokenizer, sentence, PARAM.add_index, debug=True)
		if PARAM.merge_pos:
			sent_pos = merge_pos(sent_pos, PARAM.add_index, debug=True)
		sent_pos = get_out_pos(sent_pos, debug=True)
		phrases = get_phrases(sent_pos, PARAM.window)


if __name__ == '__main__':
	start_time = time.time()
	tokenizer = get_tokenizer(PARAM.tokenizer, userdic=PARAM.dictionary)
	print_run_time(start_time, "loading")

	sentences = load_sentences()

	if PARAM.action == 'word':
		run_words(sentences, tokenizer)
	elif PARAM.action == 'coword':
		run_cowords(sentences, tokenizer)
	elif PARAM.action == 'phrase':
		run_phrases(sentences, tokenizer)

