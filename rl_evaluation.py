from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from collections import Counter

def self_bleu(sentences, is_smoothing):
    length = len(sentences)
    score = 0.0
    weight = (0.25, 0.25, 0.25, 0.25)
    smooth = SmoothingFunction()
    for i in range(length):
        for j in range(i + 1, length):
            if is_smoothing:
                #score += smooth.method4(sentence[i], sentence[j], weights=weight)
                score += sentence_bleu(sentences[i], sentences[j], weights=weight, smoothing_function=smooth.method4)
            else:
                score += sentence_bleu(sentences[i], sentences[j], weights=weight)
    score /= max(((length - 1) * length / 2), 1.0)
    return score

def parse_n_gram(sentence, n_gram):
    sentence = [''.join(sentence[i: i + n_gram]) for i in range(len(sentence) - n_gram + 1)]
    return sentence

def distinct_n(sentences, n_gram):
    lens = [len(sent) for sent in sentences]
    sentences = [parse_n_gram(sent, n_gram) for sent in sentences]

    result = Counter(sentences)

if __name__ == "__main__":
    sentences = hard_answer_seqs_list = ['去哪里玩', '你猜', '不猜', '不猜', '告诉我嘛' ]
    sentences = [' '.join(sent).split() for sent in sentences]

    # score = self_bleu(sentences, True)
    # print(score)

    print(sentences[0])
    n_gram = parse_n_gram(sentences[0], 2)
    print(n_gram)