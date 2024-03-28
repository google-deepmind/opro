# Copyright 2024 The OPRO Authors
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
r"""Optimize over the objective function of a traveling salesman problem.

Usage:

```
python optimize_tsp.py --optimizer="text-bison"
```

Note:
- When using a Google-Cloud-served model (like text-bison at
https://developers.generativeai.google/tutorials/text_quickstart), add
`--palm_api_key="<your_key>"`
- When using an OpenAI model, add `--openai_api_key="<your_key>"`
"""

import datetime
import functools
import getpass
import json
import os
import re
import sys
import itertools

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

from absl import app
from absl import flags
import google.generativeai as palm
import numpy as np
import openai

from opro import prompt_utils


_OPENAI_API_KEY = flags.DEFINE_string(
    "openai_api_key", "", "The OpenAI API key."
)

_PALM_API_KEY = flags.DEFINE_string("palm_api_key", "", "The PaLM API key.")

_OPTIMIZER = flags.DEFINE_string(
    "optimizer", "gpt-3.5-turbo", "The name of the optimizer LLM."
)

_START_ALGORITHM = flags.DEFINE_string(
    "starting_algorithm", "farthest_insertion", "The name of the starting algorithm. Select from [dp, nearest_neighbor, farthest_insertion]"
)

def main(_):
  # ============== set optimization experiment configurations ================
  num_points = 100  # number of points in TSP
  num_steps = 500  # the number of optimization steps
  max_num_pairs = 10  # the maximum number of input-output pairs in meta-prompt
  num_decimals = 0  # num of decimals for distances in meta-prompt
  num_starting_points = 5  # the number of initial points for optimization
  num_decode_per_step = 8 # the number of decoded solutions per step

  # ================ load LLM settings ===================
  optimizer_llm_name = _OPTIMIZER.value
  assert optimizer_llm_name in {
      "text-bison",
      "gpt-3.5-turbo",
      "gpt-4",
  }
  openai_api_key = _OPENAI_API_KEY.value
  palm_api_key = _PALM_API_KEY.value

  if optimizer_llm_name in {"gpt-3.5-turbo", "gpt-4"}:
    assert openai_api_key, "The OpenAI API key must be provided."
    openai.api_key = openai_api_key
  else:
    assert optimizer_llm_name == "text-bison"
    assert (
        palm_api_key
    ), "A PaLM API key is needed when prompting the text-bison model."
    palm.configure(api_key=palm_api_key)

  # =================== create the result directory ==========================
  datetime_str = (
      str(datetime.datetime.now().replace(microsecond=0))
      .replace(" ", "-")
      .replace(":", "-")
  )

  save_folder = os.path.join(
      OPRO_ROOT_PATH,
      "outputs",
      "optimization-results",
      f"tsp-o-{optimizer_llm_name}-{datetime_str}/",
  )
  os.makedirs(save_folder)
  print(f"result directory:\n{save_folder}")

  # ====================== optimizer model configs ============================
  if optimizer_llm_name.lower() == "text-bison":
    # when prompting text-bison with Cloud API
    optimizer_finetuned_palm_temperature = 1.0
    optimizer_finetuned_palm_max_decode_steps = 1024
    optimizer_finetuned_palm_batch_size = 1
    optimizer_finetuned_palm_num_servers = 1
    optimizer_finetuned_palm_dict = dict()
    optimizer_finetuned_palm_dict["temperature"] = (
        optimizer_finetuned_palm_temperature
    )
    optimizer_finetuned_palm_dict["batch_size"] = (
        optimizer_finetuned_palm_batch_size
    )
    optimizer_finetuned_palm_dict["num_servers"] = (
        optimizer_finetuned_palm_num_servers
    )
    optimizer_finetuned_palm_dict["max_decode_steps"] = (
        optimizer_finetuned_palm_max_decode_steps
    )

    call_optimizer_finetuned_palm_server_func = functools.partial(
        prompt_utils.call_palm_server_from_cloud,
        model="text-bison-001",
        temperature=optimizer_finetuned_palm_dict["temperature"],
        max_decode_steps=optimizer_finetuned_palm_dict["max_decode_steps"],
    )

    optimizer_llm_dict = {
        "model_type": optimizer_llm_name.lower(),
    }
    optimizer_llm_dict.update(optimizer_finetuned_palm_dict)
    call_optimizer_server_func = call_optimizer_finetuned_palm_server_func

  else:
    assert optimizer_llm_name in {"gpt-3.5-turbo", "gpt-4"}
    optimizer_gpt_max_decode_steps = 1024
    optimizer_gpt_temperature = 1.0

    optimizer_llm_dict = dict()
    optimizer_llm_dict["max_decode_steps"] = optimizer_gpt_max_decode_steps
    optimizer_llm_dict["temperature"] = optimizer_gpt_temperature
    optimizer_llm_dict["batch_size"] = 1
    call_optimizer_server_func = functools.partial(
        prompt_utils.call_openai_server_func,
        model=optimizer_llm_name,
        max_decode_steps=optimizer_gpt_max_decode_steps,
        temperature=optimizer_gpt_temperature,
    )

  # ====================== try calling the servers ============================
  print("\n======== testing the optimizer server ===========")
  optimizer_test_output = call_optimizer_server_func(
      "Does the sun rise from the north? Just answer yes or no.",
      temperature=1.0,
  )
  print(f"optimizer test output: {optimizer_test_output}")
  print("Finished testing the optimizer server.")
  print("\n=================================================")

  # ====================== utility functions ============================
  def evaluate_distance(x, y, trace, num_decimals):  # pylint: disable=invalid-name
    dis = 0
    try:
      for i in range(len(trace) - 1):
        id0 = trace[i]
        id1 = trace[i + 1]
        dis += np.sqrt((x[id0] - x[id1]) ** 2 + (y[id0] - y[id1]) ** 2)
    except:
      return -1
    id0 = trace[-1]
    id1 = trace[0]
    dis += np.sqrt((x[id0] - x[id1]) ** 2 + (y[id0] - y[id1]) ** 2)
    dis = np.round(dis, num_decimals) if num_decimals > 0 else int(dis)
    return dis

  def solve_tsp(x, y, num_points, num_decimals, starting_algorithm):
    if starting_algorithm == "nearest_neighbor":
      min_dis = 0
      gt_sol = [0]
      remaining_points = list(range(1, num_points))
      while len(remaining_points) > 0:
        min_p = -1
        min_cur_dis = -1
        for p in remaining_points:
          cur_dis = np.sqrt((x[p] - x[gt_sol[-1]]) ** 2 + (y[p] - y[gt_sol[-1]]) ** 2)
          if min_p == -1 or cur_dis < min_cur_dis:
            min_p = p
            min_cur_dis = cur_dis
        gt_sol.append(min_p)
        min_dis += min_cur_dis
        remaining_points.remove(min_p)
      min_dis += np.sqrt((x[0] - x[gt_sol[-1]]) ** 2 + (y[0] - y[gt_sol[-1]]) ** 2)
      min_dis = np.round(min_dis, num_decimals) if num_decimals > 0 else int(min_dis)
      return gt_sol, min_dis
    elif starting_algorithm == "farthest_insertion":
      gt_sol = [0]
      remaining_points = list(range(1, num_points))
      while len(remaining_points) > 0:
        max_p = -1
        max_cur_dis = -1
        max_cur_index = -1
        for p in remaining_points:
          min_cur_dis = -1
          min_cur_index = -1
          for index in range(1, len(gt_sol) + 1):
            new_sol = gt_sol[:index] + [p] + gt_sol[index:]
            cur_dis = evaluate_distance(x, y, new_sol, num_decimals)
            if min_cur_dis == -1 or cur_dis < min_cur_dis:
              min_cur_dis = cur_dis
              min_cur_index = index
          if max_cur_dis == -1 or min_cur_dis > max_cur_dis:
            max_p = p
            max_cur_dis = min_cur_dis
            max_cur_index = min_cur_index
        gt_sol = gt_sol[:max_cur_index] + [max_p] + gt_sol[max_cur_index:]
        remaining_points.remove(max_p)
      min_dis = evaluate_distance(x, y, gt_sol, num_decimals)
      return gt_sol, min_dis
      
    f = {(0, 1): (0, [0])}
    q = [(0, 1)]
    min_dis = -1
    gt_sol = list(range(num_points))
    while len(q) > 0:
      p, status = q[0]
      q = q[1:]
      for i in range(num_points):
        if 2 << i >> 1 & status == 0:
          new_status = status + (2 << i >> 1)
          new_dis = f[(p, status)][0] + np.sqrt((x[i] - x[p]) ** 2 + (y[i] - y[p]) ** 2)
          if (i, new_status) not in f or new_dis < f[(i, new_status)][0]:
            f[(i, new_status)] = (new_dis, f[(p, status)][1] + [i])
            if new_status == (2 << num_points >> 1) - 1:
              new_dis += np.sqrt((x[i] - x[0]) ** 2 + (y[i] - y[0]) ** 2)
              if min_dis == -1 or new_dis < min_dis:
                min_dis = new_dis
                gt_sol = f[(i, new_status)][1][:]
            elif (i, new_status) not in q:
              q.append((i, new_status))
    min_dis = np.round(min_dis, num_decimals) if num_decimals > 0 else int(min_dis)
    return gt_sol, min_dis

  def gen_meta_prompt(
      old_value_pairs_set,
      x,  # pylint: disable=invalid-name
      y,
      max_num_pairs=100,
  ):
    """Generate the meta-prompt for optimization.

    Args:
     old_value_pairs_set (set): the set of old traces.
     X (np.array): the 1D array of x values.
     y (np.array): the 1D array of y values.
     num_decimals (int): the number of decimals in the
       meta-prompt.
     max_num_pairs (int): the maximum number of exemplars in the meta-prompt.

    Returns:
      meta_prompt (str): the generated meta-prompt.
    """
    old_value_pairs = list(old_value_pairs_set)
    old_value_pairs = sorted(old_value_pairs, key=lambda x: -x[1])[
        -max_num_pairs:
    ]
    old_value_pairs_substr = ""
    for trace, dis in old_value_pairs:
      old_value_pairs_substr += f"\n<trace> {trace} </trace>\nlength:\n{dis}\n"
    meta_prompt = "You are given a list of points with coordinates below:\n"
    for i, (xi, yi) in enumerate(zip(x, y)):
      if i:
        meta_prompt += ", "
      meta_prompt += f"({i}): ({xi}, {yi})"
    meta_prompt += ".\n\nBelow are some previous traces and their lengths. The traces are arranged in descending order based on their lengths, where lower values are better.".strip()
    meta_prompt += "\n\n"
    meta_prompt += old_value_pairs_substr.strip()
    meta_prompt += "\n\n"
    meta_prompt += """Give me a new trace that is different from all traces above, and has a length lower than any of the above. The trace should traverse all points exactly once. The trace should start with '<trace>' and end with </trace>.
    """.strip()
    return meta_prompt

  def extract_string(input_string):
    start_string = "<trace>"
    end_string = "</trace>"
    if start_string not in input_string:
      return ""
    input_string = input_string[input_string.index(start_string) + len(start_string):]
    if end_string not in input_string:
      return ""
    input_string = input_string[:input_string.index(end_string)]
    parsed_list = []
    for p in input_string.split(","):
      p = p.strip()
      try:
        p = int(p)
      except:
        continue
      parsed_list.append(p)
    return parsed_list

  # ================= generate the ground truth trace =====================

  x = np.random.uniform(low=-100, high=100, size=num_points)
  y = np.random.uniform(low=-100, high=100, size=num_points)
  x = [np.round(xi, num_decimals) if num_decimals > 0 else int(xi) for xi in x]
  y = [np.round(yi, num_decimals) if num_decimals > 0 else int(yi) for yi in y]

  starting_algorithm = _START_ALGORITHM.value
  
  gt_sol, min_dis = solve_tsp(x, y, num_points, num_decimals, starting_algorithm)
  print("ground truth solution" + str(gt_sol))
  print("min distance: ", min_dis)
  gt_sol_str = ",".join([str(i) for i in gt_sol])
  point_list = range(num_points)
  init_sols = []
  while len(init_sols) < num_starting_points:
    sol = np.random.permutation(point_list)
    if sol[0] != 0:
      continue
    sol_str = ",".join([str(i) for i in sol])
    if sol_str == gt_sol_str:
      continue
    init_sols.append(list(sol))

  # ====================== run optimization ============================
  configs_dict = {
      "num_starting_points": num_starting_points,
      "num_decode_per_step": num_decode_per_step,
      "optimizer_llm_configs": optimizer_llm_dict,
      "data": {
          "ground truth solution": [",".join([str(i) for i in gt_sol])],
          "loss_at_true_values": min_dis,
          "x": list(x),
          "y": list(y),
      },
      "init_sols": [",".join([str(i) for i in sol]) for sol in init_sols],
      "num_steps": num_steps,
      "max_num_pairs": max_num_pairs,
      "num_decimals": num_decimals,
  }
  configs_json_path = os.path.join(save_folder, "configs.json")
  print(f"saving configs to\n{configs_json_path}")
  with open(configs_json_path, "w") as f:
    json.dump(configs_dict, f, indent=4)

  old_value_pairs_set = set()
  old_value_pairs_with_i_step = []  # format: [(trace, dis = f(trace), i_step)]
  meta_prompts_dict = dict()  # format: {i_step: meta_prompt}
  raw_outputs_dict = dict()  # format: {i_step: raw_outputs}

  for sol in init_sols:
    dis = evaluate_distance(x, y, sol, num_decimals)
    sol_str = ",".join([str(i) for i in sol])
    old_value_pairs_set.add((sol_str, dis))
    old_value_pairs_with_i_step.append((sol_str, dis, -1))

  print("\n================ run optimization ==============")
  print(f"initial points: {[tuple(item[:-1]) for item in old_value_pairs_set]}")
  print(f"initial values: {[item[-1] for item in old_value_pairs_set]}")
  results_json_path = os.path.join(save_folder, "results.json")
  print(f"saving results to\n{results_json_path}")

  for i_step in range(num_steps):
    print(f"\nStep {i_step}:")
    meta_prompt = gen_meta_prompt(
        old_value_pairs_set,
        x,
        y,
        max_num_pairs=max_num_pairs,
    )
    print("\n=================================================")
    print(f"meta_prompt:\n{meta_prompt}")
    meta_prompts_dict[i_step] = meta_prompt
    raw_outputs = []
    parsed_outputs = []
    while len(parsed_outputs) < num_decode_per_step:
      raw_output = call_optimizer_server_func(meta_prompt)
      for string in raw_output:
        print("\n=================================================")
        print("raw output:\n", string)
        try:
          parsed_output = extract_string(string)
          if parsed_output is not None and len(set(parsed_output)) == num_points and len(parsed_output) == num_points and parsed_output[0] == 0:
            dis = evaluate_distance(x, y, parsed_output, num_decimals)
            if dis == -1:
              continue
            parsed_outputs.append(parsed_output)
            raw_outputs.append(string)
        except:
          pass
    print("\n=================================================")
    print(f"proposed points: {parsed_outputs}")
    raw_outputs_dict[i_step] = raw_outputs

    # evaluate the values of proposed and rounded outputs
    single_step_values = []
    for trace in parsed_outputs:
      dis = evaluate_distance(x, y, trace, num_decimals)
      single_step_values.append(dis)
      trace_str = ",".join([str(i) for i in trace])
      old_value_pairs_set.add((trace_str, dis))
      old_value_pairs_with_i_step.append((trace_str, dis, i_step))
    print(f"single_step_values: {single_step_values}")
    print("ground truth solution" + str(gt_sol))
    print("min distance: ", min_dis)

    # ====================== save results ============================
    results_dict = {
        "meta_prompts": meta_prompts_dict,
        "raw_outputs": raw_outputs_dict,
        "old_value_pairs_with_i_step": old_value_pairs_with_i_step,
    }
    with open(results_json_path, "w") as f:
      json.dump(results_dict, f, indent=4)


if __name__ == "__main__":
  app.run(main)
