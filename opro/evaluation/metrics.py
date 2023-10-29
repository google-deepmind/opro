# Copyright 2023 The OPRO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Final answer parser for reasoning tasks.

The common forms of outputs to be parsed are like:
- "the answer: XXX"
- "XXX is the answer"
- "XXX is the final/right/correct answer"
"""

import dataclasses
import re
import string
from typing import Dict, List, Sequence

import immutabledict

all_letters = string.ascii_lowercase  # "abcd...xyz"
bracketed_letters_list = set([f'({l})' for l in all_letters])  # ['(a)', ...]

_WORD_TO_NUM = immutabledict.ImmutableOrderedDict({
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20,
    'thirty': 30,
    'forty': 40,
    'fifty': 50,
    'sixty': 60,
    'seventy': 70,
    'eighty': 80,
    'ninety': 90,
})
SPECIAL_NUM_CHARS = frozenset({'.', '/', ','})
# The logic for identifying patterns for the answer behind:
# First check if the primary patterns are in the string, then if not, check the
# secondary ones.
FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY = ['answer is ', 'answer: ', 'answer is: ']
FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY = ['is: ', 'are: ']
FINAL_ANSWER_AHEAD_PATTERNS = [
    ' is the correct answer',
    ' is the right answer',
    ' is the final answer',
    ' is the answer',
]
GSM8K_ANSWER = '#### '
# the Boolean symbols appeared in BBH tasks
BOOLEAN_SYMBOLS = [['false', 'true'], ['no', 'yes'], ['invalid', 'valid']]

MULTILINGUAL_QUESTION_DELIMITER = {
    'bn': {
        'Q': '\u09aa\u09cd\u09b0\u09b6\u09cd\u09a8: ',
        'A': (
            '\u09a7\u09be\u09aa\u09c7 \u09a7\u09be\u09aa\u09c7 '
            '\u0989\u09a4\u09cd\u09a4\u09b0: '
        ),
        'Direct A': '\u0989\u09a4\u09cd\u09a4\u09b0: ',
    },
    'de': {
        'Q': 'Frage: ',
        'A': 'Schritt-f\u00fcr-Schritt-Antwort: ',
        'Direct A': 'Antwort: ',
    },
    'en': {
        'Q': 'Question: ',
        'A': 'Step-by-Step Answer: ',
        'Direct A': 'Answer: ',
    },
    'es': {
        'Q': 'Pregunta: ',
        'A': 'Respuesta paso a paso: ',
        'Direct A': 'Respuesta: ',
    },
    'fr': {
        'Q': 'Question : ',
        'A': 'R\u00e9ponse \u00e9tape par \u00e9tape : ',
        'Direct A': 'R\u00e9ponse : ',
    },
    'ja': {
        'Q': '\u554f\u984c\uff1a',
        'A': '\u30b9\u30c6\u30c3\u30d7\u3054\u3068\u306e\u7b54\u3048\uff1a',
        'Direct A': '\u7b54\u3048\uff1a',
    },
    'ru': {
        'Q': '\u0417\u0430\u0434\u0430\u0447\u0430: ',
        'A': '\u041f\u043e\u0448\u0430\u0433\u043e\u0432\u043e\u0435 '
             '\u0440\u0435\u0448\u0435\u043d\u0438\u0435: ',
        'Direct A': '\u0440\u0435\u0448\u0435\u043d\u0438\u0435: ',
    },
    'sw': {
        'Q': 'Swali: ',
        'A': 'Jibu la Hatua kwa Hatua: ',
        'Direct A': 'Jibu: ',
    },
    'te': {
        'Q': '\u0c2a\u0c4d\u0c30\u0c36\u0c4d\u0c28: ',
        'A': '\u0c26\u0c36\u0c32\u0c35\u0c3e\u0c30\u0c40\u0c17\u0c3e '
             '\u0c38\u0c2e\u0c3e\u0c27\u0c3e\u0c28\u0c02: ',
        'Direct A': '\u0c38\u0c2e\u0c3e\u0c27\u0c3e\u0c28\u0c02: ',
    },
    'th': {
        'Q':
            '\u0e42\u0e08\u0e17\u0e22\u0e4c: ',
        'A':
            '\u0e04\u0e33\u0e15\u0e2d\u0e1a\u0e17\u0e35\u0e25\u0e30\u0e02\u0e31\u0e49\u0e19\u0e15\u0e2d\u0e19: ',  # pylint: disable=g-line-too-long
        'Direct A':
            '\u0e04\u0e33\u0e15\u0e2d\u0e1a\u0e17\u0e35: ',
    },
    'zh': {
        'Q': '\u95ee\u9898\uff1a',
        'A': '\u9010\u6b65\u89e3\u7b54\uff1a',
        'Direct A': '\u89e3\u7b54\uff1a',
    },
}
initial_keys = list(MULTILINGUAL_QUESTION_DELIMITER.keys())
for language in initial_keys:
  if language == 'en':
    continue
  MULTILINGUAL_QUESTION_DELIMITER[f'{language}-en'] = (
      MULTILINGUAL_QUESTION_DELIMITER['en']
  )

LANGUAGES = list(MULTILINGUAL_QUESTION_DELIMITER.keys())
NEXT_QUESTION_DELIMITERS = [
    d['Q'] for d in MULTILINGUAL_QUESTION_DELIMITER.values()
] + ['Q:']


def _is_float(s):
  try:
    float(s)
    return True
  except ValueError:
    return False


def remove_punctuation_from_string(input_string):
  output_string = input_string.translate(
      str.maketrans('', '', string.punctuation)
  )
  return output_string


def _extract_bracketed_choice_from_string(prediction):
  """Extract bracketed ABCD...XYZ choices there's exactly one bracketed choice.

  Args:
    prediction (str): the unprocessed prediction.

  Returns:
    prediction (str): the processed prediction.
  """
  prediction = prediction.lower()
  choice_in_pred_all = [item in prediction for item in bracketed_letters_list]
  if sum(choice_in_pred_all) == 1:
    prediction = re.findall(r'\(.*?\)', prediction)[0]
  return prediction


def get_normalized_prediction(prediction: str,
                              *,
                              treat_as_number: bool,
                              num_decimals: int = 0,
                              treat_as_bool: bool = False) -> str:
  """Returns a normalized prediction for use in `number_included_accuracy`.

  Args:
    prediction: The original model prediction.
    treat_as_number: Whether to treat the prediction as a number (and perform
      additional post-processing relevant to numbers, such as stripping of units
      or normalization of thousand separators, etc.).
    num_decimals: Number of decimal places to which to round the answer. Only
      applicable when treat_as_number==True.
    treat_as_bool: Whether to treat the prediction as a Boolean object. Only set
      it to True when the target is Boolean. The parser will then convert an 0/1
      answer to False/True.

  Returns:
    A normalized answer string that can be directly compared with the normalized
    golden answer in order to determine the `number_included_accuracy`.
  """

  prediction_parsed = prediction.lower().strip()

  FINAL_ANSWER_BEHIND_PATTERNS = (  # pylint: disable=invalid-name
      FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY  # pylint: disable=g-long-ternary
      if any(
          [item in prediction for item in FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY]
      )
      else FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY
  )
  DELIMITERS_FOR_ANSWER_BEHIND = (  # pylint: disable=invalid-name
      [d['A'] for d in MULTILINGUAL_QUESTION_DELIMITER.values()]
      + [GSM8K_ANSWER]
      + FINAL_ANSWER_BEHIND_PATTERNS
  )
  DELIMITERS_FOR_ANSWER_AHEAD = FINAL_ANSWER_AHEAD_PATTERNS   # pylint: disable=invalid-name

  # If the model tries to keep generating a new question, remove that additional
  # text.
  for next_question_delimiter in NEXT_QUESTION_DELIMITERS:
    prediction_parsed = prediction_parsed.split(
        next_question_delimiter.strip().lower()
    )[0]

  answer_indicated = False
  for answer_delimiter in DELIMITERS_FOR_ANSWER_BEHIND:
    if answer_delimiter.lower() in prediction_parsed:
      prediction_parsed = prediction_parsed.split(answer_delimiter.lower())[-1]
      answer_indicated = True

  for answer_delimiter in DELIMITERS_FOR_ANSWER_AHEAD:
    if answer_delimiter.lower() in prediction_parsed:
      prediction_parsed = prediction_parsed.split(answer_delimiter.lower())[0]
      answer_indicated = True

  prediction_parsed = prediction_parsed.strip()

  # Specific handling for a case that appears in one of the chain-of-thought
  # ablation experiments, where the rationale comes after final answer.
  prediction_parsed = prediction_parsed.split('this is the solution:')[0]

  # Remove trailing period.
  while prediction_parsed and prediction_parsed.endswith('.'):
    prediction_parsed = prediction_parsed[:-1]

  # Hacky fix for byte strings.
  while prediction_parsed and prediction_parsed.endswith('\''):
    prediction_parsed = prediction_parsed[:-1]

  # extract the bracketed choices: "(A) apple" -> "(a)"
  prediction_parsed = _extract_bracketed_choice_from_string(prediction_parsed)

  def _parse_without_treating_as_number(prediction_parsed):
    prediction_parsed = prediction_parsed.split('.')[0]
    return prediction_parsed

  def _parse_with_treating_as_number(prediction_parsed):
    prediction_parsed = prediction_parsed.split('=')[-1]
    for c in ['$', ',', '%', '€', '£']:
      prediction_parsed = prediction_parsed.replace(c, '')
    prediction_parsed = prediction_parsed.split(':')[0]
    prediction_parsed = prediction_parsed.strip()

    # 'eight' -> '8'.
    for word, num in _WORD_TO_NUM.items():
      if word in prediction_parsed:
        prediction_parsed = prediction_parsed.replace(word, str(num))

    corrected_answer = False

    if not corrected_answer:  # If no calculator errors were made.
      # '5600 pounds' -> '5600'; 'the 6th' -> '6'.
      if answer_indicated:
        # Take the first token that has numerical values.
        parts = prediction_parsed.split(' ')
      else:
        # Take the last token that has numerical values.
        parts = list(reversed(prediction_parsed.split(' ')))

      prediction_parsed = parts[0]  # Default
      for part in parts:
        if not part.isalpha():  # Filter out non-alphabetic tokens.
          prediction_parsed = part
          break

      # '156kgs' -> 156. '823-yard' -> 823.
      while prediction_parsed and prediction_parsed[-1].isalpha():
        prediction_parsed = prediction_parsed[:-1]
      if prediction_parsed and prediction_parsed[-1] == '-':
        prediction_parsed = prediction_parsed[:-1]

    if _is_float(prediction_parsed):
      prediction_parsed_float = round(float(prediction_parsed), num_decimals)
      prediction_parsed = '{:.{num_decimals}f}'.format(
          prediction_parsed_float, num_decimals=num_decimals)
    else:
      if re.search(r'(\d+)(?!.*\d)', prediction_parsed):
        prediction_parsed = re.search(r'(\d+)(?!.*\d)', prediction_parsed)[0]
    return prediction_parsed

  # If not expecting a Boolean result
  if not treat_as_bool:
    # If not expecting a number, then return the extracted answer as-is.
    if not treat_as_number:
      # String predictions may try to continue the sentence.
      prediction_parsed = _parse_without_treating_as_number(prediction_parsed)

    else:  # If expecting a number, do post-processing.
      prediction_parsed = _parse_with_treating_as_number(prediction_parsed)
  else:
    prediction_parsed_as_not_number = _parse_without_treating_as_number(
        prediction_parsed
    )
    prediction_parsed_as_number = _parse_with_treating_as_number(
        prediction_parsed
    )
    if not any(
        [prediction_parsed_as_not_number in item for item in BOOLEAN_SYMBOLS]
    ):
      if prediction_parsed_as_number in {'0', '1'}:
        prediction_parsed = str(bool(int(prediction_parsed_as_number))).lower()
      if prediction_parsed_as_not_number in {'0', '1'}:
        prediction_parsed = str(
            bool(int(prediction_parsed_as_not_number))
        ).lower()
    else:
      prediction_parsed = prediction_parsed_as_not_number
    # remove punctuations like ":" and then strip
    prediction_parsed = remove_punctuation_from_string(
        prediction_parsed
    ).strip()

  return prediction_parsed


@dataclasses.dataclass
class NormalizationResult:
  """Bundle of return values of get_normalized_target_and_prediction.

  Attributes:
    target: Normalized target string, suitable for direct comparison with the
      normalized prediction.
    prediction: Normalized prediction string, suitable for direct comparison
      with the normalized target.
    treat_as_number: Whether it was determined to treat the prediction as a
      number (and perform additional post-processing relevant to numbers, such
      as stripping of units or normalization of thousand separators, etc.).
    num_decimals: Number of decimal places to which it was determined to round
      the answer. Only relevant when treat_as_number==True.
  """
  target: str
  prediction: str
  treat_as_number: bool
  num_decimals: int


def get_normalized_target_and_prediction(
    target: str,
    prediction: str
    ) -> NormalizationResult:
  """Returns a normalized target and prediction for `number_included_accuracy`.

  Args:
    target: Target (i.e., golden answer). The function will automatically
      perform light normalization on the target, such as stripping off any
      answer indication prefixes like "The answer is".
    prediction: Original model prediction. The function will automatically
      normalize the prediction by stripping off trailing punctuation and any
      answer indication prefixes like "The answer is". If the target is numeric,
      will further strip units and round to the same precision as the target.

  Returns:
    The normalized target and prediction, along with related information
    indicating the types of normalization that were performed.
  """

  def _any_list_item_in_string(test_list, test_string):
    return any(item in test_string for item in test_list)

  primary_after_patterns_in_target = _any_list_item_in_string(
      FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY, target
  )
  secondary_after_patterns_in_target = _any_list_item_in_string(
      FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY, target
  )
  target = target.lower()
  if (
      primary_after_patterns_in_target
      or (
          secondary_after_patterns_in_target
          and not primary_after_patterns_in_target
      )
      or _any_list_item_in_string(FINAL_ANSWER_AHEAD_PATTERNS, target)
      or GSM8K_ANSWER in target
  ):
    if primary_after_patterns_in_target:
      target = re.split(
          r'|'.join(FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY), target
      )[-1]
    elif (
        secondary_after_patterns_in_target
        and not primary_after_patterns_in_target
    ):
      target = re.split(
          r'|'.join(FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY), target
      )[-1]
    target = re.split(r'|'.join(FINAL_ANSWER_AHEAD_PATTERNS), target)[0]
    target = target.split(GSM8K_ANSWER)[-1]
    if (
        target
        and target[-1] in [';', ',', '.']
        and _is_float(target[:-1])
    ):
      target = target[:-1]

  treat_as_number = _is_float(target)
  if treat_as_number and '.' in target:
    num_decimals = len(target.split('.')[-1])
  else:
    num_decimals = 0

  normalized_prediction = get_normalized_prediction(
      prediction,
      treat_as_number=treat_as_number,
      num_decimals=num_decimals)

  return NormalizationResult(
      target=target,
      prediction=normalized_prediction,
      treat_as_number=treat_as_number,
      num_decimals=num_decimals)


def number_included_accuracy_list(
    targets: Sequence[str],
    predictions: Sequence[str],
) -> List[bool]:
  """Returns a list of booleans for if the target is anywhere in the prediction.

  Args:
    targets: Targets (i.e., golden answers).
    predictions: Original model predictions (before normalization).
  """

  correct_list = []
  for prediction, target in zip(predictions, targets):
    normalization_result = get_normalized_target_and_prediction(
        target=target, prediction=prediction)

    # If answer is not a number, then look for exact match.
    if not normalization_result.treat_as_number:
      correct_list.append(
          normalization_result.target == normalization_result.prediction)

    else:  # If the target is a number, then compare numerically.
      correct = False  # pylint: disable=unused-variable
      try:
        prediction_parsed_float = round(
            float(normalization_result.prediction),
            normalization_result.num_decimals)
        correct = (
            abs(prediction_parsed_float - float(normalization_result.target)) <=
            1e-5)
      except ValueError:
        correct = False
      except IndexError:
        correct = False
      correct_list.append(correct)
  return correct_list


def number_included_accuracy(targets: Sequence[str],
                             predictions: Sequence[str]) -> Dict[str, float]:
  """Special accuracy for if the target is anywhere in the prediction."""

  correct_list = number_included_accuracy_list(targets, predictions)

  correct_list_with_calc = number_included_accuracy_list(
      targets, predictions)

  return {
      'accuracy':
          sum(correct_list) / len(correct_list) * 100,
      'accuracy_with_calc':
          sum(correct_list_with_calc) / len(correct_list_with_calc) * 100
  }
