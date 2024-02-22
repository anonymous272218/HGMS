
from tools.logger import logger

class VocabBase(object):
    def __init__(self):
        self._word_to_id = {}
        self._id_to_word = {}
        self._unk_token_id = -1
        self._count = 0

    def word2id(self, word):
        return self._word_to_id[word] if word in self._word_to_id else self._unk_token_id

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def __len__(self):
        return self._count

    def word_list(self):
        return self._word_to_id.keys()


class FileVocab(VocabBase):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, vocab_file, max_size, verbose=False):
        super().__init__()

        self.special_tokens = ['<pad>', '<unk>', '<s>', '</s>', '<seg>']
        self.spacial_ids = []
        for w in self.special_tokens:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self.spacial_ids.append(self._count)
            self._count += 1
        self._unk_token_id = self.word2id('<unk>')
        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r', encoding='utf8') as vocab_f:  # New : add the utf8 encoding to prevent error
            cnt = 0
            for line in vocab_f:
                cnt += 1
                pieces = line.split("\t")
                w = pieces[0]
                if w in self.special_tokens:
                    raise Exception('special_tokens shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    logger.error('Duplicated word in vocabulary file Line %d : %s' % (cnt, w))
                    continue
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    break
        if verbose:
            logger.info("Finished constructing vocab of %i total words. Last word added: %s", self._count,
                        self._id_to_word[self._count - 1])


class DictVocab(VocabBase):
    def __init__(self, vocab_dict):
        super().__init__()
        self._word_to_id = vocab_dict
        self._id_to_word = {value: key for key, value in vocab_dict.items()}
        self._unk_token_id = vocab_dict['<unk>']
        self._count = len(vocab_dict)
