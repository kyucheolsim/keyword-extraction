# written by kylesim

from konlpy.tag import Hannanum, Kkma, Komoran, Mecab, Okt

def get_tokenizer(name, userdic=None):
	if name == 'hannanum':
		tokenizer = Hannanum()
	elif name == 'kkma':
		tokenizer = Kkma()
	elif name == 'komoran':
		tokenizer = Komoran(userdic=userdic)
	elif name == 'okt':
		tokenizer = Okt()
	elif name == 'mecab':
		if userdic:
			tokenizer = Mecab(dicpath=userdic)
		else:
			tokenizer = Mecab()
	else:
		raise ValueError("%s not supported" % (name))
	return tokenizer

# run time: mecab > komoran >>> hannanum >>>> okt >>>>> kkma
# quality: mecab > komoran = kkma = okt >>> hannanum
# userdic: tokens_or_phrase[tab]POS

"""
* tokenizer: hannanum
- loading time: 0.3572251796722412
[('아버지가방에들어가', 'N'), ('이', 'J'), ('시ㄴ다', 'E')]
- run time: 2.3474743366241455

* tokenizer: kkma
- loading time: 0.03297257423400879
[('아버지', 'NNG'), ('가방', 'NNG'), ('에', 'JKM'), ('들어가', 'VV'), ('시', 'EPH'), ('ㄴ다', 'EFN')]
- run time: 13.986971378326416

* tokenizer: komoran
- loading time: 3.169203758239746
[('아버지', 'NNG'), ('가방', 'NNP'), ('에', 'JKB'), ('들어가', 'VV'), ('시', 'EP'), ('ㄴ다', 'EC')]
- run time: 0.025855064392089844

* tokenizer: okt
- loading time: 0.0032732486724853516
[('아버지', 'Noun'), ('가방', 'Noun'), ('에', 'Josa'), ('들어가신다', 'Verb')]
- run time: 7.59153938293457

* tokenizer: mecab
- loading time: 0.0061261653900146484
[('아버지', 'NNG'), ('가', 'JKS'), ('방', 'NNG'), ('에', 'JKB'), ('들어가', 'VV'), ('신다', 'EP+EC')]
- run time: 0.0016093254089355469
"""
