import numpy as np
from typing import List
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from supar import Parser
import copy

from .abst_pt import AbstPtOptimizer, AbstPt
from ..data.io import CodeTask
from ..evaluator import AbstEvaluator


# parser = Parser.load('crf-con-en')

def detokenize(tokens):
    return TreebankWordDetokenizer().detokenize(tokens)

def delete_phrase(candidate, phrase):
    if candidate.find(' ' + phrase) > 0:
        answer = candidate.replace(' ' + phrase, ' ')
    elif candidate.find(phrase + ' ') > 0:
        answer = candidate.replace(phrase + ' ', ' ')
    else:
        answer = candidate.replace(phrase, '')
    return answer

def add_phrase(candidate, phrase, after):
    if after == '':
        answer = phrase + ' ' + candidate
    else:
        if candidate.find(' ' + after) > 0:
            answer = candidate.replace(' ' + after, ' ' + after + ' ' + phrase)
        elif candidate.find(after + ' ') > 0:
            answer = candidate.replace(after + ' ', after + ' ' + phrase + ' ')
        else:
            answer = candidate.replace(after, after + phrase )
    return answer


def swap_phrases(candidate, phrase_1, phrase_2):
    # Replace phrase_1 with a placeholder
    if ' ' + phrase_1 + ' ' in candidate:
        candidate = candidate.replace(' ' + phrase_1 + ' ', ' <1> ')
    else:
        candidate = candidate.replace(phrase_1, '<1>')

    # Replace phrase_2 with a placeholder
    if ' ' + phrase_2 + ' ' in candidate:
        candidate = candidate.replace(' ' + phrase_2 + ' ', ' <2> ')
    else:
        candidate = candidate.replace(phrase_2, '<2>')

    # Swap the placeholders
    candidate = candidate.replace('<1>', phrase_2)
    candidate = candidate.replace('<2>', phrase_1)

    return candidate


def check_child(tree):
    check = False
    count = 0
    total_count = 0
    for subtree in tree:
        total_count += 1
        if type(subtree) == nltk.tree.Tree:
            if subtree.label() == '_':
                count += 1
    if count >= total_count - count: check = True

    return check

def collect_leaves(parsed_tree):
    leaves = []
    for tree in parsed_tree:
        if type(parsed_tree) != nltk.tree.Tree: continue
        if tree.label() == '_':
            leaves.append(detokenize(tree.leaves()))
            continue
        if check_child(tree): leaves.append(detokenize(tree.leaves()))
        else:
            leaves.extend(collect_leaves(tree))
    return leaves



def get_phrases(instruction): # one possible way of obtaining disjoint phrases
    phrases = []
    for sentence in sent_tokenize(instruction):
        parsed_tree = parser.predict(word_tokenize(sentence), verbose=False).sentences[0].trees[0]
        leaves = collect_leaves(parsed_tree)
        phrases.extend(leaves)
    phrases = [detokenize(word_tokenize(phrase)) for phrase in phrases if phrase not in string.punctuation or phrase == '']
    return phrases

def perform_edit(edit, base, phrase_lookup, delete_tracker):
    if edit == 'del':
        [i] = np.random.choice(list(phrase_lookup.keys()), 1)
        return delete_phrase(base, phrase_lookup[i]), [i]
    elif edit == 'swap':
        try: [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=False)
        except: [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=True)
        return swap_phrases(base, phrase_lookup[i], phrase_lookup[j]), [i, j]
    elif edit == 'sub':
        [i] = np.random.choice(list(phrase_lookup.keys()), 1)
        raise NotImplementedError
    elif edit == 'add':
        keys = list(phrase_lookup.keys())
        keys.append(-1)
        [i] = np.random.choice(keys, 1)
        if i >= 0: after = phrase_lookup[i]
        else: after = ''
        if len(delete_tracker) == 0: return base, []
        phrase = np.random.choice(delete_tracker, 1)[0]
        return add_phrase(base, phrase, after), [phrase]



class GripsOptimizer(AbstPtOptimizer):


    def __init__(
        self,
        seed_pts: List[AbstPt],
        code_llm,
        agent,
        evaluator: AbstEvaluator,
        train_data: List[CodeTask],
        valid_data: List[CodeTask],
        test_data: List[CodeTask],
        config: dict
    ):

        self.num_compose = 1
        self.num_candidates = 5
        self.patience = 3
        self.level = config['level']
        super().__init__(
            seed_pts, code_llm, agent, evaluator, train_data, valid_data, test_data, config
        )

    def get_phrase_lookup(self, base_candidate):
        if self.level == 'phrase':
            phrase_lookup = {p: phrase for p, phrase in enumerate(get_phrases(base_candidate))}
        elif self.level == 'word':
            words = word_tokenize(base_candidate)
            words = [w for w in words if w not in string.punctuation or w != '']
            phrase_lookup = {p: phrase for p, phrase in enumerate(words)}
        elif self.level == 'sentence':
            sentences = sent_tokenize(base_candidate)
            phrase_lookup = {p: phrase for p, phrase in enumerate(sentences)}
        elif self.level == 'span':
            phrases = []
            for sentence in sent_tokenize(base_candidate):
                spans_per_sentence = np.random.choice(range(2, 5))  # split sentence into 2, 3, 4, 5 chunks
                spans = np.array_split(word_tokenize(sentence), spans_per_sentence)
                spans = [detokenize(s) for s in spans]
                phrases.extend(spans)
            phrase_lookup = {p: phrase for p, phrase in enumerate(phrases)}
        else:
            raise ValueError()
        return phrase_lookup

    def create_tmp_pt(self, inst):
        pt = copy.deepcopy(self.seed_pts[0])
        pt.instruction = inst
        return pt

    def optimize_pt(self):

        instruction = self.seed_pts[0].instruction

        valid_scores_list = []
        train_scores_list = []
        best_pt_ist = []
        edit_operations = ['del', 'swap', 'add']
        operations_tracker = []
        base_candidate = detokenize(word_tokenize(instruction))
        base_score = self.score_pt(self.create_tmp_pt(base_candidate), self.train_data)
        delete_tracker = []
        patience_counter = 1

        for iter_i in range(self.num_steps):
            valid_score = self.score_pt(self.create_tmp_pt(base_candidate), self.valid_data)
            print(f"{iter_i}: Valid score {valid_score}")
            valid_scores_list.append(valid_score)
            phrase_lookup = self.get_phrase_lookup(base_candidate)
            self.update_edit_operations(edit_operations, delete_tracker)
            edits = self.generate_edits(edit_operations, self.num_candidates, self.num_compose)

            candidates, deleted, added = self.generate_candidates(
                edits, base_candidate, phrase_lookup, delete_tracker
            )

            scores = [self.score_pt(self.create_tmp_pt(candidate), self.train_data) for candidate in candidates]

            base_candidate, base_score, patience_counter = self.update_best_candidate(
                candidates, scores, base_candidate, base_score, delete_tracker, deleted, added, patience_counter
            )
            train_scores_list.append(base_score)
            best_pt_ist.append(self.create_tmp_pt(base_candidate))
            if patience_counter > self.patience:
                break
        return {
            "best_pt": self.create_tmp_pt(base_candidate),
            "best_pt_list": best_pt_ist,
            "valid_scores_list": valid_scores_list,
            "train_scores_list": train_scores_list
        }

    def update_edit_operations(self, edit_operations, delete_tracker):
        if delete_tracker:
            if 'add' not in edit_operations:
                edit_operations.append('add')
        else:
            if 'add' in edit_operations:
                edit_operations.remove('add')

    def generate_edits(self, edit_operations, num_candidates, num_compose):
        if num_compose == 1:
            return np.random.choice(edit_operations, num_candidates)
        return [np.random.choice(edit_operations, num_compose) for _ in range(num_candidates)]

    def generate_candidates(self, edits, base_candidate, phrase_lookup, delete_tracker):
        candidates, deleted, added = [], {}, {}

        for edit in edits:
            if isinstance(edit, str):
                candidate, indices = perform_edit(edit, base_candidate, phrase_lookup, delete_tracker)
                candidates.append(candidate)
                self.track_edit_operations(edit, candidate, indices, deleted, added, phrase_lookup)
            else:
                candidate, composed_deletes, composed_adds = self.apply_composed_edits(edit, base_candidate, delete_tracker)
                candidates.append(candidate)
                if composed_deletes:
                    deleted[candidate] = composed_deletes
                if composed_adds:
                    added[candidate] = composed_adds

        return candidates, deleted, added

    def track_edit_operations(self, edit, candidate, indices, deleted, added, phrase_lookup):
        if edit == 'del':
            deleted[candidate] = [phrase_lookup[indices[0]]]
        elif edit == 'add' and indices:
            added[candidate] = indices

    def apply_composed_edits(self, edit, base_candidate, delete_tracker):
        old_candidate = base_candidate
        composed_deletes, composed_adds = [], []

        for op in edit:
            phrase_lookup = self.get_phrase_lookup(old_candidate)
            new_candidate, indices = perform_edit(op, old_candidate, phrase_lookup, delete_tracker)
            if op == 'del':
                composed_deletes.append(phrase_lookup[indices[0]])
            elif op == 'add' and indices:
                composed_adds.append(indices[0])
            old_candidate = new_candidate

        return old_candidate, composed_deletes, composed_adds

    def update_best_candidate(self, candidates, scores, base_candidate, base_score, delete_tracker, deleted, added,
                              patience_counter):
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        if best_score > base_score:
            base_candidate = candidates[best_idx]
            base_score = best_score
            patience_counter = 1

            self.update_delete_tracker(base_candidate, delete_tracker, deleted, added)
            base_candidate = detokenize(word_tokenize(base_candidate))
        else:
            patience_counter += 1

        return base_candidate, base_score, patience_counter

    def update_delete_tracker(self, base_candidate, delete_tracker, deleted, added):
        if base_candidate in added:
            for chunk in added[base_candidate]:
                if chunk in delete_tracker:
                    delete_tracker.remove(chunk)
        if base_candidate in deleted:
            delete_tracker.extend(deleted[base_candidate])


