import numpy as np
import WordMetrics
import WordMatching as wm
from string import punctuation



def matchSampleAndRecordedWords(real_text, handwritten_transcript):
        words_estimated = handwritten_transcript.split()

        if real_text is None:
            words_real = handwritten_transcript.split()
        else:
            words_real = real_text.split()

        mapped_words, mapped_words_indices = wm.get_best_mapped_words(
            words_estimated, words_real)

        real_and_transcribed_words = []
        #real_and_transcribed_words_ipa = []
        for word_idx in range(len(words_real)):
            if word_idx >= len(mapped_words)-1:
                mapped_words.append('-')
            real_and_transcribed_words.append(
                (words_real[word_idx], mapped_words[word_idx]))
            
        return real_and_transcribed_words, mapped_words_indices


def getHandwritingAccuracy(real_and_transcribed_words) -> float:
    total_mismatches = 0.
    number_of_char = 0.
    current_words_handwritten_accuracy = []
    for pair in real_and_transcribed_words:
        real_without_punctuation = removePunctuation(pair[0]).lower()
        number_of_word_mismatches = WordMetrics.edit_distance_python(
            real_without_punctuation, removePunctuation(pair[1]).lower())
        total_mismatches += number_of_word_mismatches
        number_of_char_in_word = len(real_without_punctuation)
        number_of_char += number_of_char_in_word

        current_words_handwritten_accuracy.append(float(
            number_of_char_in_word-number_of_word_mismatches)/number_of_char_in_word*100)

    percentage_of_correct_handwriting = (
        number_of_char-total_mismatches)/number_of_char*100

    return np.round(percentage_of_correct_handwriting), current_words_handwritten_accuracy


def removePunctuation(word: str) -> str:
    return ''.join([char for char in word if char not in punctuation])

def matchSampleAndRecordedWords(real_text, handwritten_transcript):
    words_estimated = handwritten_transcript.split()

    if real_text is None:
        words_real = handwritten_transcript.split()
    else:
        words_real = real_text.split()

    mapped_words, mapped_words_indices = wm.get_best_mapped_words(words_estimated, words_real)
    print("mapped_words", mapped_words)
    print("mapped_words_indices", mapped_words_indices)
    real_and_transcribed_words = []
    #real_and_transcribed_words_ipa = []
    for word_idx in range(len(words_real)):
        if word_idx >= len(mapped_words)-1:
            mapped_words.append('-')
        real_and_transcribed_words.append(
            (words_real[word_idx], mapped_words[word_idx]))
        
    #real_and_transcribed_words_ipa.append((self.ipa_converter.convertToPhonem(words_real[word_idx]),
    #                                        self.ipa_converter.convertToPhonem(mapped_words[word_idx])))
    return real_and_transcribed_words, mapped_words_indices







