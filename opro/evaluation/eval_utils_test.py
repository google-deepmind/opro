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
"""Tests for metrics."""

import os
import sys

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

from absl.testing import absltest
from absl.testing import parameterized
from opro.evaluation import eval_utils


class UtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("remove_punc", "Let's go.", "Lets go<PERIOD>"),
  )
  def test_remove_punc(self, input_sentence, output_sentence):
    self.assertEqual(
        output_sentence,
        eval_utils.remove_punctuation_from_string(input_sentence),
    )

  @parameterized.named_parameters(
      ("empty_filename", "", "<NO INSTRUCTION>"),
      ("filename_with_linebreak", "a\nb", "ab"),
      ("filename_with_punc", "Let's go.", "Lets go<PERIOD>"),
      ("filename_with_linebreak_and_punc", "a:\nb ?", "ab <QUESTION>"),
  )
  def test_instruction_to_filename(self, instruction, filename):
    self.assertEqual(filename, eval_utils.instruction_to_filename(instruction))

  @parameterized.named_parameters(
      ("no_change_for_well_formatted_sentence", "Let's go.", "Let's go."),
      ("white_space_before_and_afterwards_removed", " Let's go. ", "Let's go."),
      ("capitalize_first_letter", "let's go.", "Let's go."),
      ("do_not_touch_question_mark", "Let's go?", "Let's go?"),
      ("do_not_touch_exclamation", "Let's go!", "Let's go!"),
  )
  def test_polish_sentence(self, original_sentence, expected_polished_sentence):
    self.assertEqual(
        expected_polished_sentence,
        eval_utils.polish_sentence(original_sentence),
    )

  @parameterized.named_parameters(
      ("get_index_from_symbol_0", "b", 1),
      ("get_index_from_symbol_1", "(c)", 2),
      ("get_index_from_symbol_2", "(D)", 3),
  )
  def test_get_index_from_symbol(self, answer, expected_result):
    self.assertEqual(expected_result, eval_utils._get_index_from_symbol(answer))

  @parameterized.named_parameters(
      (
          "get_answer_text_example",
          (
              "From which direction does the sun rise in the morning? (A) west"
              " (B) east (C) north (D) south (E) northwest"
          ),
          "(E)",
          "northwest",
      ),
  )
  def test_get_answer_text(
      self, input_text, true_answer_symbol, expected_result
  ):
    self.assertEqual(
        expected_result,
        eval_utils._get_answer_text(input_text, true_answer_symbol),
    )

  @parameterized.named_parameters(
      ("accuracy_of_symbol_without_brackets_correct", "(A)", "a", "", 1),
      ("accuracy_of_symbol_without_brackets_wrong", "(A)", "b", "", 0),
      ("accuracy_of_symbol_with_brackets_correct", "(A)", "(a)", "", 1),
      ("accuracy_of_symbol_with_brackets_wrong", "(A)", "(b)", "", 0),
      (
          "accuracy_of_text_match_correct",
          "(B)",
          "east",
          (
              "From which direction does the sun rise in the morning? (A) west"
              " (B) east (C) north (D) south"
          ),
          1,
      ),
      (
          "accuracy_of_text_with_bracket_and_punc_match_correct",
          "(B)",
          "b/c! ",
          (
              "This is a dummy (x) question: (A) a/b$ (B) b/c! (C) c/d (D) d/a"
          ),
          1,
      ),
      (
          "accuracy_of_text_match_wrong",
          "(B)",
          "west",
          (
              "From which direction does the sun rise in the morning? (A) west"
              " (B) east (C) north (D) south"
          ),
          0,
      ),
      (
          "accuracy_of_symbol_match_with_text_correct",
          "(B)",
          "b",
          (
              "From which direction does the sun rise in the morning? (A) west"
              " (B) east (C) north (D) south"
          ),
          1,
      ),
      (
          "accuracy_of_symbol_match_with_text_wrong",
          "(B)",
          "a",
          (
              "From which direction does the sun rise in the morning? (A) west"
              " (B) east (C) north (D) south"
          ),
          0,
      ),
  )
  def test_accuracy_of_individuals(
      self, true_answer, pred_answer, input_text, expected_result
  ):
    self.assertEqual(
        expected_result,
        eval_utils._get_accuracy(true_answer, pred_answer, input_text),
    )

  @parameterized.named_parameters(
      ("accuracy_of_list_without_text", "A", ["A", "A", "A", "B"], "", 0.75),
      (
          "accuracy_of_list_with_test",
          "(B)",
          ["A", "east", "b", "(B)", "(D)"],
          (
              "From which direction does the sun rise in the morning? (A) west"
              " (B) east (C) north (D) south"
          ),
          0.6,
      ),
  )
  def test_accuracy_of_list(
      self, true_answer, pred_answer_list, input_text, expected_result
  ):
    self.assertEqual(
        expected_result,
        eval_utils.get_accuracy_of_list(
            true_answer, pred_answer_list, input_text
        ),
    )

  @parameterized.named_parameters(
      (
          "accuracy_of_symbol_match",
          "B",
          "(b)",
          (
              "This is a (dummy) question. (A) west (B) east west (C) north (D)"
              " south\nWhat's the answer in (A)(B)(C)(D)?"
          ),
          1,
      ),
      ("accuracy_of_answer_match_with_punctuations", "Yes", ":yes", "", 1),
      ("accuracy_of_boolean_match_on_text_1", "Yes", "yes", "", 1),
      ("accuracy_of_boolean_match_on_text_2", "True", "true", "", 1),
      ("accuracy_of_boolean_match_on_meaning_1", "Yes", "true", "", 1),
      ("accuracy_of_boolean_match_on_meaning_2", "Yes", "false", "", 0),
      ("accuracy_of_boolean_match_on_meaning_3", "Yes", "1", "", 1),
      ("accuracy_of_boolean_match_on_meaning_4", "Invalid", "true", "", 0),
      ("accuracy_of_boolean_match_on_meaning_5", "Invalid", "false", "", 1),
      ("accuracy_of_boolean_match_on_meaning_6", "Invalid", "1", "", 0),
      (
          "accuracy_of_symbol_not_match",
          "B",
          "(a)",
          (
              "This is a (dummy) question. (A) west (B) east west (C) north (D)"
              " south\nWhat's the answer in (A)(B)(C)(D)?"
          ),
          0,
      ),
      (
          "accuracy_of_text_exact_match",
          "B",
          "east west",
          (
              "This is a (dummy) question. (A) west (B) east west (C) north (D)"
              " south\nWhat's the answer in (A)(B)(C)(D)?"
          ),
          1,
      ),
      (
          "accuracy_of_text_exact_match_case_2",
          "A",
          "west",
          (
              "This is a (dummy) question. (A) west (B) east west (C) north (D)"
              " south\nWhat's the answer in (A)(B)(C)(D)?"
          ),
          1,
      ),
      (
          "accuracy_of_text_included",
          "B",
          "east west is reported",
          (
              "This is a (dummy) question. (A) west (B) east west (C) north (D)"
              " south\nWhat's the answer in (A)(B)(C)(D)?"
          ),
          1,
      ),
      (
          "accuracy_of_text_included_case_2",
          "A",
          "west is reported",
          (
              "This is a (dummy) question. (A) west (B) east west (C) north (D)"
              " south\nWhat's the answer in (A)(B)(C)(D)?"
          ),
          1,
      ),
      (
          "accuracy_of_text_included_with_punc_and_space_correct_1",
          "A",
          ": west",
          (
              "This is a (dummy) question. (A) west (B) east west (C) north (D)"
              " south\nWhat's the answer in (A)(B)(C)(D)?"
          ),
          1,
      ),
      (
          "accuracy_of_text_included_with_punc_and_space_correct_2",
          "A",
          ": west is reported",
          (
              "This is a (dummy) question. (A) west (B) east west (C) north (D)"
              " south\nWhat's the answer in (A)(B)(C)(D)?"
          ),
          1,
      ),
      (
          "accuracy_of_text_included_with_punc_and_space_not_correct",
          "A",
          ": east",
          (
              "This is a (dummy) question. (A) west (B) east west (C) north (D)"
              " south\nWhat's the answer in (A)(B)(C)(D)?"
          ),
          0,
      ),
      (
          "accuracy_of_text_not_included_case_1",
          "B",
          "west is reported",
          (
              "This is a (dummy) question. (A) west (B) east west (C) north (D)"
              " south\nWhat's the answer in (A)(B)(C)(D)?"
          ),
          0,
      ),
      (
          "accuracy_of_text_not_included_case_2",
          "A",
          "east west is reported",
          (
              "This is a (dummy) question. (A) west (B) east west (C) north (D)"
              " south\nWhat's the answer in (A)(B)(C)(D)?"
          ),
          0,
      ),
  )
  def test_get_accuracy(
      self, true_answer, pred_answer, input_text, expected_result
  ):
    self.assertEqual(
        expected_result,
        eval_utils._get_accuracy(true_answer, pred_answer, input_text),
    )


if __name__ == "__main__":
  absltest.main()
