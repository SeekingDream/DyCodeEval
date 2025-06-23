import copy
import random
import torch
import string
import nltk
from transformers import BertTokenizer, pipeline
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from typing import List
from .abst_method import AbstTaskMutation


def contains_only_characters(input_string: str):
    if len(input_string) == 0:
        return False
    for char in input_string:
        if not char.isalpha():
            return False
    return True


class _DescriptionMutation(AbstTaskMutation):
    def __init__(self, prompt, test, entry_point):
        super().__init__(prompt, test, entry_point)

        # self.tree_tokenizer = TreebankWordTokenizer()
        # self.detokenizer = TreebankWordDetokenizer()
        bert_model_name = 'bert-base-uncased'
        self.device = torch.device('cuda')
        self.berttokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.unmasker = pipeline('fill-mask', model='bert-base-uncased')

    def mutate(self, language):
        func_entry, comments, demo = self.split_desc_testcases(language)
        assert self.combine_desc_testcases(language, func_entry, comments, demo) == self.prompt
        if language == 'py' or 'cs':
            new_comments = self.mutate_py(comments)
        else:
            raise NotImplementedError
        return self.combine_desc_testcases(language, func_entry, new_comments, demo)

    def mutate_py(self, desc):
        raise NotImplementedError


class CharacterMutation(_DescriptionMutation):
    def __init__(self, prompt, test, entry_point):
        super().__init__(prompt, test, entry_point)
        self.num_of_perturb = 3  # delat is the number of token/words allowed to be modified
        self.letters_to_insert = string.ascii_letters

    def mutate_py(self, comments):
        tokens = comments.split(" ")

        possible_mutate_indexL = []
        # collect the token index for substitution
        for idx, word in enumerate(tokens):
            if contains_only_characters(word) == False:
                continue
            possible_mutate_indexL.append((idx, word))
        bert_new_sentences = list()

        # generate similar setences using Bert
        if possible_mutate_indexL:
            bert_new_sentences = self.character_mutation(tokens, possible_mutate_indexL)
        if len(bert_new_sentences) == 0:
            new_comments = comments
        else:
            new_comments = random.choice(bert_new_sentences)
        return new_comments

    def _get_random_letter(self):
        """Helper function that returns a random single letter from the English
        alphabet that could be lowercase or uppercase."""
        return random.choice(self.letters_to_insert)

    def _get_neighbor_swap_words(self, word):
        """Returns a list containing all possible words with 1 pair of
        neighboring characters swapped."""

        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 0
        end_idx = (len(word) - 1)

        if start_idx >= end_idx:
            return []

        for i in range(start_idx, end_idx):
            candidate_word = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
            candidate_words.append(candidate_word)

        return candidate_words

    def _get_character_insert_words(self, word):
        """Returns returns a list containing all possible words with 1 random
        character inserted."""
        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 0
        end_idx = len(word)

        if start_idx >= end_idx:
            return []

        for i in range(start_idx, end_idx):
            candidate_word = word[:i] + self._get_random_letter() + word[i:]
            candidate_words.append(candidate_word)

        return candidate_words

    def _get_character_delete_words(self, word):
        """Returns returns a list containing all possible words with 1 letter
        deleted."""
        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 0
        end_idx = len(word)

        if start_idx >= end_idx:
            return []
        for i in range(start_idx, end_idx):
            candidate_word = word[:i] + word[i + 1 :]
            candidate_words.append(candidate_word)

        return candidate_words

    def _get_homoglyph_replace_words(self, word):
        """Returns a list containing all possible words with 1 character
        replaced by a homoglyph."""
        candidate_words = []
        homos = {
            "-": "˗",
            "9": "৭",
            "8": "Ȣ",
            "7": "𝟕",
            "6": "б",
            "5": "Ƽ",
            "4": "Ꮞ",
            "3": "Ʒ",
            "2": "ᒿ",
            "1": "l",
            "0": "O",
            "'": "`",
            "a": "ɑ",
            "b": "Ь",
            "c": "ϲ",
            "d": "ԁ",
            "e": "е",
            "f": "𝚏",
            "g": "ɡ",
            "h": "հ",
            "i": "і",
            "j": "ϳ",
            "k": "𝒌",
            "l": "ⅼ",
            "m": "ｍ",
            "n": "ո",
            "o": "о",
            "p": "р",
            "q": "ԛ",
            "r": "ⲅ",
            "s": "ѕ",
            "t": "𝚝",
            "u": "ս",
            "v": "ѵ",
            "w": "ԝ",
            "x": "×",
            "y": "у",
            "z": "ᴢ",
        }
        for i in range(len(word)):
            if word[i] in homos:
                repl_letter = homos[word[i]]
                candidate_word = word[:i] + repl_letter + word[i + 1 :]
                candidate_words.append(candidate_word)

        return candidate_words

    def character_mutation(self, tokens, possible_mutate_indexL):
        base_tokens = copy.deepcopy(tokens)
        new_sentences = []
        for _ in range(50):
            sampled_num = min(len(possible_mutate_indexL), self.num_of_perturb)
            samples = random.sample(possible_mutate_indexL, sampled_num)
            mask_indices = [k[0] for k in samples]
            for masked_index in mask_indices:
                current_token = tokens[masked_index]
                tokens[masked_index] = self._mutate(current_token)
            new_sentence = " ".join(tokens)
            new_sentences.append(new_sentence)

            tokens = copy.deepcopy(base_tokens)
        return new_sentences

    def _mutate(self, word):
        func_list = [
            self._get_neighbor_swap_words,
            self._get_character_insert_words,
            self._get_character_delete_words,
            self._get_homoglyph_replace_words
        ]
        mutants = []
        for func in func_list:
            candidates = func(word)
            mutants.extend(candidates)
        if len(mutants) == 0:
            return word
        return random.choice(mutants)


class TokenMutation(_DescriptionMutation):
    def __init__(self, prompt, test, entry_point):
        super().__init__(prompt, test, entry_point)
        self.num_of_perturb = 3

    def mutate_py(self, comments: str) -> str:
        tokens = comments.split(" ")
        pos_inf = nltk.tag.pos_tag(tokens)

        # the elements in the lists are tuples <index of token, pos tag of token>
        bert_masked_indexL = list()

        # collect the token index for substitution
        for idx, (word, tag) in enumerate(pos_inf):
            if contains_only_characters(word) == False:
                continue
            # substitute the nouns and adjectives; you could easily substitue more words by modifying the code here
            if tag.startswith('NN') or tag.startswith('JJ'):
                tagFlag = tag[:2]
                # we do not perturb the first and the last token because BERT's performance drops on for those positions
                if idx != 0 and idx != len(tokens) - 1:
                    bert_masked_indexL.append((idx, tagFlag))

        bert_new_sentences = list()

        # generate similar setences using Bert
        if bert_masked_indexL:
            bert_new_sentences = self.perturbBert(tokens, bert_masked_indexL)

        if len(bert_new_sentences) == 0:
            new_desc = comments
        else:
            new_desc = random.choice(bert_new_sentences)
        return new_desc

    def perturbBert(self, tokens: List[str], masked_indexL: List):
        base_tokens = copy.deepcopy(tokens)
        # self.bertmodel, self.num_of_perturb,
        new_sentences = list()

        for _ in range(10):
            sampled_num = min(len(masked_indexL),  self.num_of_perturb)
            samples = random.sample(masked_indexL, sampled_num)
            mask_indices = [k[0] for k in samples]
            low_tokens = [x.lower() for x in tokens]
            for masked_index in mask_indices:
                low_tokens[masked_index] = '[MASK]'
            new_str = " ".join(low_tokens)
            unmask_str = self.unmasker(new_str)
            try:
                filed_token = [d[0]['token_str'] for d in unmask_str]
            except:
                filed_token = [d['token_str'] for d in unmask_str]
            for index, token in zip(mask_indices, filed_token):
                tokens[index] = token
            new_sentence = " ".join(tokens)
            tokens = copy.deepcopy(base_tokens)
            new_sentences.append(new_sentence)
        return new_sentences

        #         # for each idx, use Bert to generate k (i.e., num) candidate tokens
        # for (masked_index, tagFlag) in masked_indexL:
        #     original_word = tokens[masked_index]
        #
        #     low_tokens = [x.lower() for x in tokens]
        #     low_tokens[masked_index] = '[MASK]'
        #     try:
        #         indexed_tokens = self.berttokenizer.convert_tokens_to_ids(low_tokens)
        #         tokens_tensor = torch.tensor([indexed_tokens])
        #         tokens_tensor = tokens_tensor.to(self.device)
        #         prediction = self.bertmodel(tokens_tensor)
        #     except KeyError as error:
        #         print('skip a sentence. unknown token is %s' % error)
        #         break
        #
        #
        #
        #     # try whether all the tokens are in the vocabulary
        #     try:
        #         indexed_tokens = self.berttokenizer.convert_tokens_to_ids(low_tokens)
        #         tokens_tensor = torch.tensor([indexed_tokens])
        #         tokens_tensor = tokens_tensor.to(self.device)
        #         prediction = self.bertmodel(tokens_tensor)
        #
        #     # skip the sentences that contain unknown words
        #     # another option is to mark the unknow words as [MASK]; we skip sentences to reduce fp caused by BERT
        #     except KeyError as error:
        #         print('skip a sentence. unknown token is %s' % error)
        #         break
        #
        #     # get the similar words
        #     topk_Idx = torch.topk(prediction[0][0, masked_index], self.num_of_perturb)[1].tolist()
        #     topk_tokens = self.berttokenizer.convert_ids_to_tokens(topk_Idx)
        #
        #     # remove the tokens that only contains 0 or 1 char (e.g., i, a, s)
        #     # this step could be further optimized by filtering more tokens (e.g., non-english tokens)
        #     topk_tokens = list(filter(lambda x: len(x) > 1, topk_tokens))
        #
        #     # generate similar sentences
        #     for t in topk_tokens:
        #         if any(char in invalidChars for char in t):
        #             continue
        #         tokens[masked_index] = t
        #         new_pos_inf = nltk.tag.pos_tag(tokens)
        #
        #         # only use the similar sentences whose similar token's tag is still NN or JJ
        #         if (new_pos_inf[masked_index][1].startswith(tagFlag)):
        #             # new_sentence = self.detokenizer.detokenize(tokens)
        #             new_sentence = " ".join(tokens)
        #             new_sentences.append(new_sentence)
        #
        #     tokens[masked_index] = original_word


class TaskGenerator:
    def __init__(self):
        pass


