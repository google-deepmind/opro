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
"""The utility functions for evaluation."""

import functools
import hashlib
import json
from multiprocessing import dummy as mp  # multithreading
import os
import re
import string
import sys
import time

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

import numpy as np
from opro.evaluation import metrics
import pandas as pd

# the Boolean symbols appeared in BBH tasks
BOOLEAN_SYMBOLS = [["false", "true"], ["no", "yes"], ["invalid", "valid"]]

all_lowercase_letters = string.ascii_lowercase  # "abcd...xyz"
bracketed_lowercase_letters_set = set(
    [f"({l})" for l in all_lowercase_letters]
)  # {"(a)", ...}
bracketed_uppercase_letters_set = set(
    [f"({l.upper()})" for l in all_lowercase_letters]
)  # {"(a)", ...}


def read_jsonl(filepath):
  """Read the jsonl file (AQuA raw data)."""
  with open(filepath, "r", encoding="utf-8") as fh:
    return [json.loads(line) for line in fh.readlines() if line]


def remove_punctuation_from_string(input_string, is_filename=True):
  """Remove punctuations from string to comply with filename requirements."""
  # remove punctuations other than "!", "?", "."
  if is_filename:
    punctuation_subset_str = (
        string.punctuation.replace("!", "").replace("?", "").replace(".", "")
    )
    output_string = input_string.translate(
        str.maketrans("", "", punctuation_subset_str)
    )
    # replace punctuations "!", "?", "." with indicating letters
    output_string = (
        output_string.replace("!", "<EXCLAMATION>")
        .replace("?", "<QUESTION>")
        .replace(".", "<PERIOD>")
    )
  else:
    output_string = input_string.translate(
        str.maketrans("", "", string.punctuation)
    )
  return output_string


def instruction_to_filename(instruction, md5_hashing=True):
  """Convert an instruction string to filename."""
  if md5_hashing:
    m = hashlib.md5()
    m.update(instruction.encode("ascii"))
    filename = m.hexdigest()
  else:
    # remove punctuations and line break, and give a name to the empty string
    filename = instruction.replace("\n", "")
    filename = remove_punctuation_from_string(repr(filename))
    filename = filename if filename else "<NO INSTRUCTION>"
  return filename


def polish_sentence(sentence, add_ending_punc=False):
  """Standardize the sentence to English syntax.

  This is used in prompt optimization to keep track of previously evaluated
  instructions, and is NOT used to create the filename for individual
  instruction results.

  Args:
    sentence (str): the original sentence.
    add_ending_punc (bool): whether to add an ending punctuation.

  Returns:
    sentence (str): the polished sentence.
  """
  sentence = sentence.strip()
  if sentence:
    sentence = sentence.replace("**", "")
    if len(sentence) > 1:
      sentence = (
          sentence[0].upper() + sentence[1:]
      )  # capitalize the first letter
    if add_ending_punc and not (
        sentence.endswith(".")
        or sentence.endswith("?")
        or sentence.endswith("!")
    ):
      sentence += "."
  return sentence


# pylint: disable=invalid-name
def _split_by_Q(sentence):
  """Split the response and only keep the part before the first "Q:"."""
  return sentence.split("Q:")[0].strip()


def _format_mmlu_example(data, idx, include_question=True):
  """Generate the question part of the MMLU prompt.

  Modified from https://github.com/hendrycks/test/blob/master/evaluate.py.

  Args:
    data (pandas.DataFrame): the comma-delimited MMLU raw data with no index or
      header, and with columns: question, Choice A, Choice B, Choice C, Choice
      D, true answer in ABCD
    idx (int): the index of the question in data
    include_question (bool): whether to include the final question sentence in
      the question. The include_question argument is set to True by default, and
      for now there is no option to change it in gen_prompt.

  Returns:
    prompt (str): the generated question.
  """
  choices = ["(A)", "(B)", "(C)", "(D)"]  # MMLU questions only have 4 choices
  prompt = data.iloc[idx, 0]
  k = data.shape[1] - 2
  for j in range(k):
    prompt += "\n{} {}".format(choices[j], data.iloc[idx, j + 1])
  if include_question:
    prompt += "\nWhat's the answer in (A) (B) (C) (D)?"
  return prompt


def _format_aqua_example(data, idx, include_question=True):
  """Generate the question part of the AQuA prompt."""
  question = data[idx]["question"]
  options = ["(" + item for item in data[idx]["options"]]
  for item in options:
    question += f"\n{item}"
  if include_question:
    question += "\nWhat's the answer in (A) (B) (C) (D) (E)?"
  return question


def gen_prompt(
    data,
    instruction,
    idx,
    include_qa=True,
    instruction_pos="Q_begin",
    dataset_name="mmlu",
):
  """Generate a prompt from the available exemplars and the given instruction.

  The MMLU case was modified from
  https://github.com/hendrycks/test/blob/master/evaluate.py.

  Args:
    data (pandas.DataFrame or list or json): the input-output pairs.
      pandas.DataFrame for MMLU or GSM8K, list for BBH, json for Multiarith.
    instruction (str): the instruction.
    idx (int): the index of the exemplar in the data list.
    include_qa (bool): whether to include "Q:" and "A:" formats in the prompt.
    instruction_pos (str): where to put the instruction, one of {'before_Q',
      'Q_begin', 'Q_end', 'A_begin'}.
    dataset_name (str): one of {"mmlu", "bbh", "gsm8k"}.

  Returns:
    prompt (str): the generated prompt.
  """
  dataset_name = dataset_name.lower()
  assert dataset_name in {
      "mmlu",
      "bbh",
      "gsm8k",
      "multiarith",
      "aqua",
  }, (
      "The lower-case dataset name must be one of mmlu, bbh, gsm8k, multiarith,"
      " or aqua."
  )
  assert instruction_pos in {
      "before_Q",
      "Q_begin",
      "Q_end",
      "A_begin",
  }, (
      "The instruction position should be either before the question, or at the"
      " beginning of the question, at the end of the question, or at the"
      " beginning of the answer."
  )
  if dataset_name == "mmlu":
    question = _format_mmlu_example(data, idx)
  elif dataset_name == "bbh":
    question = data[idx]["input"]
  elif dataset_name == "gsm8k":
    question = data.iloc[idx, 0]
  elif dataset_name == "multiarith":
    question = data[idx]["sQuestion"].strip()
  else:
    assert dataset_name == "aqua"
    question = _format_aqua_example(data, idx)

  prompt = ""
  if include_qa:  # when "Q:" and "A:" are present in the prompt
    if instruction_pos == "before_Q":
      if instruction:
        prompt += instruction + "\n"
      prompt += "Q: " + question
      prompt += "\n\nA:"
    elif instruction_pos == "Q_begin":
      if instruction:
        prompt += "Q: " + instruction + "\n"
      else:
        prompt += "Q: "
      prompt += question
      prompt += "\n\nA:"
    elif instruction_pos == "Q_end":
      prompt += "Q: " + question
      if instruction:
        prompt += "\n" + instruction + "\n\nA:"
      else:
        prompt += "\n\nA:"
    else:
      assert instruction_pos == "A_begin"
      prompt += f"Q: {question}\n\n"
      prompt += "A:"
      if instruction:
        prompt += f" {instruction}"
  else:  # when there're no "Q:" and "A:" in the prompt
    assert instruction_pos in {"Q_begin", "Q_end"}
    if instruction_pos == "Q_begin":
      if instruction:
        prompt += instruction + "\n"
      prompt += question
    else:  # instruction_pos == "Q_end"
      prompt += question
      if instruction:
        prompt += "\n" + instruction
  return prompt


def fetch_true_answer(data, idx, dataset_name):
  """Fetch the true answer of the dataset at the idx'th position."""
  dataset_name = dataset_name.lower()
  assert dataset_name in {
      "mmlu",
      "bbh",
      "gsm8k",
      "multiarith",
      "aqua",
  }, (
      "The lower-case dataset name must be one of mmlu, bbh, gsm8k, multiarith,"
      " or aqua."
  )
  if dataset_name == "mmlu":
    return data.iloc[idx, -1]
  elif dataset_name == "bbh":
    return data[idx]["target"]
  elif dataset_name == "gsm8k":
    return data.iloc[idx, 1]
  elif dataset_name == "multiarith":
    return int(data[idx]["lSolutions"][0])
  else:
    assert dataset_name == "aqua"
    return data[idx]["correct"]


def _get_index_from_symbol(answer):
  """Get the index from the letter symbols A, B, C, D, to extract answer texts.

  Args:
    answer (str): the string of answer like "(B)".

  Returns:
    index (int): how far the given choice is from "a", like 1 for answer "(B)".
  """
  answer = str(answer).lower()
  # extract the choice letter from within bracket
  if answer in bracketed_lowercase_letters_set:
    answer = re.findall(r"\(.*?\)", answer)[0][1]
  index = ord(answer) - ord("a")
  return index


def _get_answer_text(input_text, answer_symbol):
  """Get the text of an answer from the symbol of a multiple choice question.

  Args:
    input_text (str): the case-sensitive input or prompt that contains choice
      letters and texts, like "From which direction does the sun rise in the
      morning? (A) west (B) east (C) north (D) south". Must contain consecutive
      upper-case bracketed letters like (A) (B) (C) (D).
    answer_symbol (str): the symbol of the true answer, like "(B)" in the above
      example.

  Returns:
    answer_text (str): the text of the trueanswer, like "east" in the
    above example.
  """
  # The choice_text_list may contain the answer part "A: xxx", but it doesn't
  # matter because the index returned by _get_index_from_symbol() is unlikely
  # to be that of "A: xxx"
  re_split_string = (
      "".join([rf"\({l.upper()}\)|" for l in all_lowercase_letters]) + "A:"
  )
  choice_text_list = [
      item.strip().lower() for item in re.split(re_split_string, input_text)
  ][1:]
  choice_text_list = [
      re.split("\n", item)[0] for item in choice_text_list
  ]  # remove the '\n' from the text of the last choice
  # Note the input_text needs to have choice symbols in consecutive order, like
  # "(A) ... (B) ... (C) ... (D) ... (E) ..."
  answer_text = choice_text_list[_get_index_from_symbol(answer_symbol)]
  return answer_text


def _prompting_to_get_raw_answers(
    prompts,
    call_server_func,
    server_index=1,
    max_retry=1,
    sleep_time=60,
    verbose=False,
):
  """Prompt to get the output to the input prompt.

  Args:
    prompts (str or list): a prompt string or a list of strings (in which each
      element is a prompt).
    call_server_func (function): the name of the function that calls the
      inference server.
    server_index (int): (PaLM only) the index of the server to prompt.
    max_retry (int): the maximum number of retries.
    sleep_time (int): the number of seconds to sleep before a retry.
    verbose (bool): whether to print out progress information.

  Returns:
    outputs (list): a list of strings, each being the output of the
    corresponding prompt. The output is a list even if the input is a list.
  """
  outputs = []
  for i in range(int(max_retry + 1)):
    if i > 0:
      if verbose:
        print(
            f"retry {i}/{max_retry} after sleeping for {sleep_time:.0f} seconds"
        )
      time.sleep(sleep_time)
    try:
      outputs = call_server_func(prompts, server_index=server_index)
    except:  # pylint: disable=bare-except
      continue
    break
  assert (
      outputs
  ), "No prompting output after all retries, indicating possible server outage."
  return outputs


def _get_accuracy(
    true_answer, pred_answer, input_text="", treat_include_as_correct=False
):
  """Get the accuracy of a prediction.

  Args:
    true_answer (str/int/float): the true answer, like "(B)".
    pred_answer (str/int/float): the answer given in one decode, like "(A)".
    input_text (str): the case-sensitive input or prompt that contains choice
      letters and texts, like "From which direction does the sun rise in the
      morning? (A) west (B) east (C) north (D) south". Must contain consecutive
      upper-case bracketed letters like (A) (B) (C) (D).
    treat_include_as_correct (bool): whether to treat the answer as correct when
      true_answer is included in pred_answer.

  Returns:
    accuracy (int): 1 or 0, indicating the answer is right or wrong.
  """
  # the comments below follow the example in the above docstring
  true_answer = str(true_answer).lower()  # "(b)"
  pred_answer = str(pred_answer).lower()  # "(a)"
  true_answer_included_in_pred_answer = true_answer in pred_answer
  if input_text:  # for multiple choice questions
    if true_answer in all_lowercase_letters:
      true_answer = f"({true_answer})"
    if pred_answer in all_lowercase_letters:
      pred_answer = f"({pred_answer})"
    if true_answer not in bracketed_lowercase_letters_set:
      return 0
    true_answer_text = _get_answer_text(
        input_text=input_text, answer_symbol=true_answer
    ).lower()  # 'east'
    all_symbols_raw = np.unique(re.findall(r"\([A-Z]\)", input_text))
    all_symbols = []  # to be ['(A)', '(B)', '(C)', '(D)']
    for item in sorted(list(bracketed_uppercase_letters_set)):
      if item in all_symbols_raw:
        all_symbols.append(item)
      else:
        break
    other_answer_texts_list = []  # ['west', 'north', 'south']
    for symbol in all_symbols:
      if _get_index_from_symbol(symbol) != _get_index_from_symbol(true_answer):
        other_answer_texts_list.append(
            _get_answer_text(input_text=input_text, answer_symbol=symbol)
        )
  else:
    other_answer_texts_list = []
    true_answer_text = ""
  # extract the choice symbol from within bracket
  if true_answer in bracketed_lowercase_letters_set:
    true_answer = re.findall(r"\(.*?\)", true_answer)[0][1]  # 'b'
  if pred_answer in bracketed_lowercase_letters_set:
    pred_answer = re.findall(r"\(.*?\)", pred_answer)[0][1]  # 'a'
  result_exact_match = (pred_answer == true_answer) or (
      remove_punctuation_from_string(pred_answer, is_filename=False).strip()
      == remove_punctuation_from_string(true_answer, is_filename=False).strip()
  )  # False
  is_choice_text_exact_match = bool(input_text) and (
      pred_answer == true_answer_text
      or remove_punctuation_from_string(pred_answer).strip() == true_answer_text
  )

  def _text_in_list_not_in_target(text_list, target):
    return all([item not in target for item in text_list])

  def _target_not_in_any_of_text_list(target, text_list):
    return all([target not in text for text in text_list])

  is_true_choice_text_included_and_other_choice_text_excluded = (
      bool(input_text)
      and true_answer_text in pred_answer
      and (  # pylint: disable=g-long-ternary
          _text_in_list_not_in_target(
              other_answer_texts_list, pred_answer.replace(true_answer_text, "")
          )
          if _target_not_in_any_of_text_list(
              true_answer_text, other_answer_texts_list
          )
          else _text_in_list_not_in_target(other_answer_texts_list, pred_answer)
      )
  )
  # If the true answer is a Boolean symbol, check "Boolean match".
  is_boolean_match = False
  if any([true_answer in item for item in BOOLEAN_SYMBOLS]):
    boolean_type_index = np.where(
        [true_answer in item for item in BOOLEAN_SYMBOLS]
    )[0][0]
    true_answer_as_true_or_false_str = str(
        bool(
            np.where(
                np.array(BOOLEAN_SYMBOLS[boolean_type_index]) == true_answer
            )[0][0]
        )
    ).lower()
    if pred_answer in {"0", "1"}:
      pred_answer = str(bool(int(pred_answer))).lower()
    is_boolean_match = (
        pred_answer == true_answer_as_true_or_false_str
        or pred_answer.strip() == true_answer_as_true_or_false_str.strip()
    )

  accuracy = int(
      result_exact_match
      or is_choice_text_exact_match
      or is_true_choice_text_included_and_other_choice_text_excluded
      or is_boolean_match
  )
  if treat_include_as_correct:
    accuracy = int(bool(accuracy) or true_answer_included_in_pred_answer)
  return accuracy

  # Alternatively, we may only check if the true_answer string is in the bag of
  # words of pred_answer, to avoid false negatives like when
  # true_answer == '(A)' and pred_answer == '(A) <some explanations>'.
  # The code would be "if true_answer.lower() in pred_answer.lower().split():".
  # However, this may incur false positives, so we don't adopt it for now.


def get_accuracy_of_list(
    true_answer,
    pred_answer_list,
    input_text="",
    treat_include_as_correct=False,
):
  """Get the accuracy of a list of predictions.

  Args:
    true_answer (str or list): the true answer, like 'A' or ['yes'].
    pred_answer_list (list): the list of answers given in multiple decodes, like
      ['A', 'A', 'B', 'C', 'C']. Each entry is the answer in one decode.
    input_text (str): for multiple choice questions, the raw input or prompt
      that contains choice letters and texts, like "From which direction does
      the sun rise in the morning? (A) west (B) east (C) north (D) south"
    treat_include_as_correct (bool): whether to treat the answer as correct when
      true_answer is included in pred_answer.

  Returns:
    accuracy (float): the accuracy of the list, like 0.4 for the above example.
  """
  # pylint: disable=g-long-lambda
  assert not isinstance(true_answer, list)
  accuracy_list = list(
      map(
          lambda x: _get_accuracy(
              true_answer=true_answer,
              pred_answer=x,
              input_text=input_text,
              treat_include_as_correct=treat_include_as_correct,
          ),
          pred_answer_list,
      )
  )
  return np.average(accuracy_list)


def evaluate_single_instruction(
    data,
    instruction,
    eval_index_all,
    batch_size,
    call_server_func,
    dataset_name,
    num_servers,
    extract_final_answer_by_prompting_again,
    instruction_pos,
    is_multiple_choice,
    include_qa=True,
    evaluate_in_parallel=True,
    num_decodes=1,
    max_retry=5,
    sleep_time=60,
    prediction_treat_as_number=False,
    prediction_treat_as_bool=False,
    prediction_num_decimals=0,
    is_gpt_model=False,
    verbose=False,
):
  r"""Evaluate a single instruction on the given indices of the given data.

  Args:
    data (list): the input-output pairs.
    instruction (str): the instruction.
    eval_index_all (list or np.ndarray): a list or tuple of indices that we'll
      evaluate on.
    batch_size (int): the batch size in model serving.
    call_server_func (function): the name of the function that calls the
      inference server.
    dataset_name (str): "mmlu" or "bbh".
    num_servers (int): the number of inference servers.
    extract_final_answer_by_prompting_again (bool): We can often get
      well-formatted answer when the model has been instruction-finetuned;
      otherwise, we may need to prompt again with "So the final answer is" added
      to better extract the final answer for final parsing.
    instruction_pos (str): where to put the instruction, one of {'before_Q',
      'Q_begin', 'Q_end', 'A_begin'}.
    is_multiple_choice (bool or list[bool]): whether the questions are multiple
      choice. Boolean indicates the status for the entire task; a list of
      Boolean indicates the status of each question.
    include_qa (bool): whether to include "Q:" and "A:" formats in the prompt.
    evaluate_in_parallel (bool): whether to evaluate the instructions in
      parallel with multithreading. Should be set to False when prompting GPT
      models.
    num_decodes (int): the number of decodes in model serving.
    max_retry (int): the maximum number of retries.
    sleep_time (int): the number of seconds to sleep before a retry.
    prediction_treat_as_number (bool or 'adaptive'): if bool, the
      treat_as_number argument in metrics.get_normalized_prediction(); if
      'adaptive', will treat prediction as number if and only if the
      corresponding true answer is numeric.
    prediction_treat_as_bool (bool): the treat_as_bool argument in
      metrics.get_normalized_prediction().
    prediction_num_decimals (int): the num_decimals argument in
      metrics.get_normalized_prediction().
    is_gpt_model (bool): Whether the scorer model is a GPT model. This flag
      exists because GPT models often output the final answer in "\boxed{}".
    verbose (bool): whether to print out progress information.

  Returns:
    detailed_results_df (pandas.DataFrame): the prompts, results, true answers
    and accuracies. Columns are ['raw_prompt', 'raw_answer', 'parsed_answer',
    'true_answer', 'accuracy'].
  """
  assert prediction_treat_as_number == "adaptive" or isinstance(
      prediction_treat_as_number, bool
  )
  assert instruction_pos in {
      "before_Q",
      "Q_begin",
      "Q_end",
      "A_begin",
  }, (
      "The instruction position should be either before the question, or at the"
      " beginning of the question, at the end of the question, or at the"
      " beginning of the answer."
  )
  num_eval_examples = len(eval_index_all)
  assert type(is_multiple_choice) in {bool, list}, (
      "is_multiple_choice must be a Boolean variable or a list of Boolean"
      " variables"
  )
  if isinstance(is_multiple_choice, bool):
    is_multiple_choice = [is_multiple_choice] * num_eval_examples
  else:
    assert (
        len(is_multiple_choice) == num_eval_examples
    ), "is_multiple_choice must have the same length as eval_index_all"

  true_answers = [
      fetch_true_answer(data, idx=idx, dataset_name=dataset_name)
      for idx in eval_index_all
  ]

  # generate raw prompts
  raw_prompts_flattened = []
  for i in range(num_eval_examples):
    raw_prompt = gen_prompt(
        data,
        instruction=instruction,
        idx=eval_index_all[i],
        include_qa=include_qa,
        instruction_pos=instruction_pos,
        dataset_name=dataset_name,
    )
    raw_prompts_flattened.append(raw_prompt)

  if evaluate_in_parallel:

    def _prompt_a_list_in_parallel(
        raw_prompts_flattened,
        num_servers,
        call_server_local_func,
    ):
      num_examples = len(raw_prompts_flattened)
      raw_prompts_grouped_by_batch_size = []
      raw_prompts_single_batch = []
      i = 0
      while i < num_examples:
        raw_prompt = raw_prompts_flattened[i]
        raw_prompts_single_batch.append(raw_prompt)
        i += 1
        if i % batch_size == 0:
          raw_prompts_grouped_by_batch_size.append(raw_prompts_single_batch)
          raw_prompts_single_batch = []
      if raw_prompts_single_batch:
        raw_prompts_grouped_by_batch_size.append(raw_prompts_single_batch)

      server_indices = [
          i % num_servers + 1
          for i in range(len(raw_prompts_grouped_by_batch_size))
      ]  # [1, 2, ..., num_servers, 1, 2, ..., num_servers, 1, 2, ...]

      p1 = mp.Pool(num_servers)
      # pylint: disable=g-complex-comprehension
      r = [
          p1.apply_async(
              _prompting_to_get_raw_answers,
              args=[
                  raw_prompts_single_batch,
                  call_server_local_func,
                  server_index,
                  max_retry,
                  sleep_time,
                  verbose,
              ],
          )
          for raw_prompts_single_batch, server_index in list(
              zip(raw_prompts_grouped_by_batch_size, server_indices)
          )
      ]
      p1.close()
      p1.join()

      raw_answers = []
      for i in range(len(raw_prompts_grouped_by_batch_size)):
        # when there're multiple decodes, only retain the first answer
        raw_answers += r[i].get()[:batch_size]
      return raw_answers

    # first round of prompting to get raw answers
    raw_answers = _prompt_a_list_in_parallel(
        raw_prompts_flattened=raw_prompts_flattened,
        num_servers=num_servers,
        call_server_local_func=call_server_func,
    )
  else:  # no parallelism in first round
    raw_answers = [
        call_server_func(prompt)[0] for prompt in raw_prompts_flattened
    ]

  if verbose:
    print("first round of prompting finished")

  # prompt again to better extract answers
  if extract_final_answer_by_prompting_again:
    raw_prompts_flattened_second_round = list(
        map(
            lambda a, b: a + " " + _split_by_Q(b),
            raw_prompts_flattened,
            raw_answers,
        )
    )
    raw_prompts_flattened_second_round = [
        item + " " + "So the final answer is"
        for item in raw_prompts_flattened_second_round
    ]

    # second round of prompting to extract final answer
    # We only need a small max_decode_steps because the answer usually shows up
    # at the very beginning of the output. The decode length can't be too small
    # though, because on some GSM8K questions the second-round answers include
    # some calculations before arriving at the final answer
    if evaluate_in_parallel:
      # pylint: disable=undefined-variable
      raw_answers_second_round = _prompt_a_list_in_parallel(
          raw_prompts_flattened=raw_prompts_flattened_second_round,
          num_servers=num_servers,
          call_server_local_func=functools.partial(
              call_server_func, max_decode_steps=50
          ),
      )
    else:
      raw_answers_second_round = [
          call_server_func(prompt, max_decode_steps=50)[0]
          for prompt in raw_prompts_flattened_second_round
      ]
    if verbose:
      print("second round of prompting finished")

  if verbose:
    print(
        "extracting final prediction with"
        f" treat_as_number={prediction_treat_as_number},"
        f" treat_as_bool={prediction_treat_as_bool}, and"
        f" num_decimals={prediction_num_decimals}"
    )

  # Based on specific formats of the second-round answers, the function below
  # extracts the corresponding texts for parsing. Here're roles of all parts:
  # .strip(":") - following "the answer is", some answers have ":" at the
  # beginning
  # .strip() - some answers have "\n" or blank spaces at the beginning, or have
  # "\n" after ":"
  # .split("\n")[0] - extract the texts before the first "\n\n" after the above
  # stripping
  # .split("Q:")[0] - extract the texts before "Q:" after the above stripping
  def _extract_second_round_answer_for_parsing(ans):
    return ans.strip(":").strip().split("\n")[0].split("Q:")[0]

  raw_answers_to_parse = (
      list(  # pylint: disable=g-long-ternary
          map(
              _extract_second_round_answer_for_parsing, raw_answers_second_round
          )
      )
      if extract_final_answer_by_prompting_again
      else raw_answers
  )

  if prediction_treat_as_number == "adaptive":
    true_answer_is_numeric = [item.isnumeric() for item in true_answers]
    prediction_treat_as_number_list = true_answer_is_numeric.copy()
  else:
    assert isinstance(prediction_treat_as_number, bool)
    prediction_treat_as_number_list = [prediction_treat_as_number] * len(
        true_answers
    )

  def _parse_prediction(
      x, is_gpt_model, treat_as_number, num_decimals, treat_as_bool
  ):
    if is_gpt_model and r"\boxed" in x:
      return re.findall(r"\\boxed{(.*?)}", x)[0]
    else:
      return metrics.get_normalized_prediction(
          x,
          treat_as_number=treat_as_number,
          num_decimals=num_decimals,
          treat_as_bool=treat_as_bool,
      )

  # pylint: disable=g-long-lambda
  choices = list(
      map(
          lambda x, y: _parse_prediction(
              x,
              is_gpt_model,
              y,
              prediction_num_decimals,
              prediction_treat_as_bool,
          ),
          raw_answers_to_parse,
          prediction_treat_as_number_list,
      )
  )
  if not extract_final_answer_by_prompting_again:
    choices = [
        _extract_second_round_answer_for_parsing(item) for item in choices
    ]

  accuracies = []
  for i, _ in enumerate(eval_index_all):
    treat_include_as_correct = not prediction_treat_as_number_list[i]
    input_text = raw_prompts_flattened[i] if is_multiple_choice[i] else ""
    accuracy = get_accuracy_of_list(
        true_answer=true_answers[i],
        pred_answer_list=choices[
            int(num_decodes * i) : int(num_decodes * (i + 1))
        ],
        input_text=input_text,
        treat_include_as_correct=treat_include_as_correct,
    )
    accuracies.append(accuracy)

  detailed_results_df = pd.DataFrame(
      list(
          zip(
              eval_index_all,
              raw_prompts_flattened,
              raw_answers,
              choices,
              true_answers,
              accuracies,
          )
      ),
      columns=[
          "index_in_raw_dataset",
          "raw_prompt",
          "raw_answer",
          "parsed_answer",
          "true_answer",
          "accuracy",
      ],
  )
  if extract_final_answer_by_prompting_again:
    detailed_results_df.insert(
        3, "raw_prompt_second_round", raw_prompts_flattened_second_round
    )
    detailed_results_df.insert(
        4, "raw_answer_second_round", raw_answers_second_round
    )

  detailed_results_df.set_index("index_in_raw_dataset", inplace=True)
  return detailed_results_df


# functions to read BBH data
# modified from http://google3/third_party/py/cascades/examples/tasks/bbh.py;rcl=501965439 # pylint: disable=line-too-long


def get_bbh_task_names(bbh_root_folder_path):
  files = os.listdir(bbh_root_folder_path)
  task_names = [f.split(".json")[0] for f in files]
  task_names = [f for f in task_names if "." not in f]
  return task_names


def load_bbh_task_data(
    task_name: str,
    base_dir: str,
    qa_format: bool = True,
):
  """Load BBH raw data from disk.

  The data is available at https://github.com/suzgunmirac/BIG-Bench-Hard.

  Args:
    task_name (str): which bbh task to load
    base_dir (str): the directory containing json files for bbh.
    qa_format (bool): whether to prepend "Q:" and "A:" to raw input and target,
      respectively

  Returns:
    data (list): a list of examples, each example is a dict {'input':
    <question_string>, 'target': <answer_string>}
  """

  if task_name not in get_bbh_task_names(base_dir):
    raise ValueError(
        f"Task {task_name} not a valid bbh task.  Consult `get_task_names()`"
        " for a list of valid tasks."
    )

  task_loc = f"{base_dir}/{task_name}.json"
  with open(task_loc, "r") as f:
    data = json.loads(f.readlines()[0])["examples"]

  if qa_format:
    formatted_examples = []
    for d in data:
      # uses BIG-bench formatting
      formatted_examples.append(
          {"input": f"{d['input']}", "target": f"{d['target']}"}
      )
    data = formatted_examples

  return data
