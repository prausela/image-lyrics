from typing import Any
import math

def _element_count(
        elements: list
) -> dict[Any, int]:
    element_map = dict()
    for element in elements:
        if element not in element_map:
            element_map[element] = 0

        element_map[element] += 1
    return element_map

def _count_matching(
        hypothesis: list,
        reference_bag: dict[Any, int]
) -> int:
    matching = 0
    for hypothesis_element in hypothesis:
        if hypothesis_element in reference_bag:
            matching += 1

            reference_bag[hypothesis_element] -= 1
            if reference_bag[hypothesis_element] <= 0:
                reference_bag.pop(hypothesis_element)

    return matching

def _build_n_grams(
        sentence: list[str],
        n: int
) -> list[str]:
    if n == 1:
        return sentence
    
    n_grams = []
    for i in range(len(sentence) - n + 1):
        n_gram = []
        for j in range(n):
            n_gram.append(sentence[i+j])
        n_grams.append(" ".join(n_gram))
    return n_grams

def calculate_simple_bleu_single_ref(
        hypothesis: list[str],
        reference: list[str],
        n: int = 1
) -> float:
    if n <= 0:
        raise ValueError('n must be greater than 0')
    
    hypothesis_len = len(hypothesis)
    if hypothesis_len < n:
        raise ValueError(f'n must be less than hypothesis length. n = {n}, hypothesis_len = {hypothesis_len}')
    
    hypothesis = _build_n_grams(hypothesis, n)
    reference = _build_n_grams(reference, n)

    reference_bag = _element_count(reference)
    matching_words = _count_matching(hypothesis, reference_bag)
    return matching_words / len(hypothesis)

def calculate_simple_bleu_multi_ref(
        hypothesis: list[str],
        references: list[list[str]],
        n: int = 1
) -> float:
    bleu_sum = 0
    for reference in references:
        bleu_sum += calculate_simple_bleu_single_ref(hypothesis, reference, n)
    return bleu_sum / len(references)

def _geometric_mean(
        values: list[float]
) -> float:
    log_value_sum = 0
    for value in values:
        log_value_sum += math.log(value)

    return math.exp((1/len(values))*log_value_sum)

def calculate_combined_bleu_single_ref(
        hypothesis: list[str],
        reference: list[str],
        max_n: int = 4
) -> float:
    bleu_scores = []
    for n in range(1, max_n + 1):
        bleu_score = calculate_simple_bleu_single_ref(hypothesis, reference, n)
        if bleu_score == 0:
            return 0.0
        
        bleu_scores.append(bleu_score)
    return _geometric_mean(bleu_scores)

def calculate_combined_bleu_multi_ref(
        hypothesis: list[str],
        references: list[list[str]],
        max_n: int = 4
) -> float:
    bleu_scores = []
    for n in range(1, max_n + 1):
        bleu_score = calculate_simple_bleu_multi_ref(hypothesis, references, n)
        if bleu_score == 0:
            return 0.0
        
        bleu_scores.append(bleu_score)
    return _geometric_mean(bleu_scores)

def _calculate_brevity_penalty(
        hypothesis: list[str],
        references: list[list[str]]
) -> float:
    if len(references) <= 0:
        raise ValueError('must provide reference sentences')
    
    hypothesis_len = len(hypothesis)
    best_match_len = None
    best_match_difference = float('inf')
    for reference in references:
        reference_len = len(reference)
        difference = abs(reference_len - hypothesis_len)
        if best_match_len is None or difference < best_match_difference:
            best_match_len = reference_len
            best_match_difference = difference

    if hypothesis_len > best_match_len:
        return 1
    
    return math.exp(1 - (best_match_len / hypothesis_len))

def calculate_blue_score(
        hypothesis: list[str],
        references: list[list[str]],
        max_n: int = 4
) -> float:
    bp = _calculate_brevity_penalty(hypothesis, references)
    return bp * calculate_combined_bleu_multi_ref(hypothesis, references, max_n)