import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

from eval.task import BaseBenchmark

# Modified version of hendrycks_math with additional instruction to mark the solution with \\boxed
# https://github.com/mlfoundations/evalchemy/blob/e70a45e41cb2ada273d6bb98e75dba303ec31f8b/eval/chat_benchmarks/AMC23/eval_instruct.py#L15
PROMPT = """Problem: {problem}\nMark your solution with \\boxed\nAnswer:"""


class AIME25Benchmark(BaseBenchmark):
    """
    AIME25 Benchmark for evaluating the math reasoning of LLMs.
    Link: https://huggingface.co/datasets/zwhe99/aime25

    Follows the evaluation logic of hendrycks_math answer extraction.
    """

    def __init__(
        self,
        data_file: str = "eval/chat_benchmarks/AIME25/data/aime25.json",
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        max_gen_length_stage1: int = 4096, # Max tokens for the first stage (reasoning)
        unthink_string: str = '</think>\n\n### :white_check_mark: Final Answer:\n\n', # String to inject
        extra_tokens_after_unthink: int = 128, # Tokens to generate after unthink
        enable_thinking: bool = True, # Whether to use the <think> prompt format
        temperature: float = 0.7, # Generation temperature
        do_sample: bool = False, # Whether to use sampling (False for deterministic math)

    ):
        """
        Initialize AIME25 benchmark.

        Args:
            data_file: File containing the AIME25 dataset (id, problem, reference_solution, expected_answer, source)
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
            system_instruction: Optional system instruction string.
            max_gen_length_stage1: Max tokens for the initial reasoning stage.
            unthink_string: The string to inject after the first stage.
            extra_tokens_after_unthink: Number of tokens to generate after injecting unthink_string.
            enable_thinking: If True, uses the chat template with thinking enabled.
            temperature: Generation temperature.
            do_sample: Whether to use sampling during generation.
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
        self.debug = debug
        # self.max_new_tokens = 32768
        self.seed = seed
        self.n_repeat = 1
        self.max_gen_length_stage1 = max_gen_length_stage1
        self.unthink_string = unthink_string
        self.extra_tokens_after_unthink = extra_tokens_after_unthink
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self.do_sample = do_sample


    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and temporary directory,
            or None for non-primary ranks
        """
        hf_model = model.model
        tokenizer = model.tokenizer
        device = model.device # Get device from the lm-eval model wrapper

        # Tokenize the unthink string and don't add special tokens
        unthink_tokens = tokenizer.encode(
            self.unthink_string,
            return_tensors='pt',
            add_special_tokens=False
        ).to(device)


        examples = self.load_questions()
        if self.debug:
             examples = examples[:2] # Limit examples if debugging

        all_outputs_repeated = [] # Store outputs for each repetition

        for i in range(self.n_repeat):
            self.logger.info(f"Starting generation repetition {i+1}/{self.n_repeat}")
            current_seed = [s + i for s in self.seed] # Adjust seed per repetition if needed
            torch.manual_seed(current_seed[0])
            np.random.seed(current_seed[1])

            outputs_current_repetition = []
            for idx, example in enumerate(examples):
                messages = []
                if self.system_instruction:
                     messages.append({'role': 'system', 'content': self.system_instruction})
                messages.append(
                    {"role": "user", "content": PROMPT.format(problem=example["problem"])}
                )

                # Tokenize using the chat template logic
                try:
                    tokenized_chat = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=self.enable_thinking,
                        return_tensors="pt"
                    ).to(device)
                except Exception as e:
                    self.logger.error(f"Error tokenizing chat for example {idx}: {e}")
                    self.logger.error(f"Messages: {messages}")
                    outputs_current_repetition.append(f"ERROR: Tokenization failed - {e}")
                    continue

                # Stage 1 generation
                try:
                    generate_ids_stage1 = hf_model.generate(
                        tokenized_chat,
                        max_new_tokens=self.max_gen_length_stage1,
                        do_sample=self.do_sample,
                        temperature=self.temperature if self.do_sample else None,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                except Exception as e:
                     self.logger.error(f"Error during Stage 1 generation for example {idx}: {e}")
                     outputs_current_repetition.append(f"ERROR: Stage 1 generation failed - {e}")
                     continue # Skip this example


                # --- Concatenate with Unthink Tokens ---
                # Ensure batch dim matches (should be 1 here)
                if generate_ids_stage1.shape[0] != unthink_tokens.shape[0]:
                   unthink_tokens_batch = unthink_tokens.repeat(generate_ids_stage1.shape[0], 1)
                else:
                    unthink_tokens_batch = unthink_tokens
                input_ids_stage2 = torch.cat([generate_ids_stage1, unthink_tokens_batch], dim=-1)

                # Stage 2 Generation
                try:
                    generate_ids_stage2 = hf_model.generate(
                        input_ids_stage2,
                        max_new_tokens=self.extra_tokens_after_unthink,
                        do_sample=self.do_sample,
                        temperature=self.temperature if self.do_sample else None,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                except Exception as e:
                     self.logger.error(f"Error during Stage 2 generation for example {idx}: {e}")
                     outputs_current_repetition.append(f"ERROR: Stage 2 generation failed - {e}")
                     continue # Skip this example


                # Decode output from stage 2
                final_text = tokenizer.decode(
                    generate_ids_stage2[0], # Assuming batch size is 1
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False # Preserve spaces for formatting
                )

                # Remove the input prompt part from the final text
                input_length = tokenized_chat.shape[1]
                completion_tokens = generate_ids_stage2[0, input_length:]
                completion_text = tokenizer.decode(
                    completion_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                # Store full generated text
                outputs_current_repetition.append(final_text)

            all_outputs_repeated.append(outputs_current_repetition)

        # We need to transpose the results: examples should have a list of outputs, one per repetition
        final_outputs_per_example = list(zip(*all_outputs_repeated))

        if len(final_outputs_per_example) != len(examples):
             self.logger.error(f"Mismatch in number of examples ({len(examples)}) and generated outputs ({len(final_outputs_per_example)}). Check for errors during generation.")
             padded_outputs = []
             for i in range(len(examples)):
                  if i < len(final_outputs_per_example):
                      padded_outputs.append(final_outputs_per_example[i])
                  else:
                      # Add error strings for the missing example for all repetitions
                      padded_outputs.append(["ERROR: Generation missing"] * self.n_repeat)
             final_outputs_per_example = padded_outputs


        for example, outputs in zip(examples, final_outputs_per_example):
            example["model_outputs"] = list(outputs) # List of outputs, one per repetition
            example["model_answers"] = [self.extract_answer(o) for o in outputs]

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions."""

        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)

        # Calculate accuracy for each repetition
        all_results = []
        for i in range(self.n_repeat):
            correct_count = 0
            for example in examples:
                if i < len(example["model_answers"]):
                    model_ans = example["model_answers"][i]
                    expected_ans = str(example["answer"])
                    if isinstance(model_ans, str) and not model_ans.startswith("ERROR:") :
                         if is_equiv(expected_ans, model_ans):
                              correct_count += 1
                    elif not isinstance(model_ans, str):
                         self.logger.warning(f"Unexpected type for model_answer[{i}] in example {example.get('id', 'N/A')}: {type(model_ans)}")
                else:
                    self.logger.warning(f"Missing model answer for repetition {i+1} in example {example.get('id', 'N/A')}")

            accuracy = correct_count / num_questions if num_questions > 0 else 0
            all_results.append(
                {
                    "repetition": i + 1,
                    "num_total": num_questions,
                    "num_solved": correct_count,
                    "accuracy": accuracy,
                }
            )

        # Calculate overall statistics
        solved_avg = np.mean([result["num_solved"] for result in all_results])
        accuracy_avg = np.mean([result["accuracy"] for result in all_results])
        accuracy_std_err = np.std([result["accuracy"] for result in all_results]) / np.sqrt(self.n_repeat) if self.n_repeat > 0 else 0

        results.update(
            {
                "num_total": num_questions,
                "solved_avg": solved_avg,
                "run_stats": all_results,
                "accuracy_avg": accuracy_avg,
                "accuracy_std_err": accuracy_std_err,
                "num_repeat": self.n_repeat,
            }
        )

        return results

    def load_questions(self) -> List[Dict[str, str]]:
        """Load AIME25 questions from the data file."""
        with open(self.data_file, "r") as f:
            questions = [json.loads(x) for x in f]
        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions

    def extract_answer(self, output: str) -> str:
        """Extract the final answer from a model-generated solution, which is expected to be in the format of \boxed{answer}.

        Uses the same logic as hendrycks_math.

        Args:
            output (str): Model-generated solution text

        Returns:
            str: Extracted final answer. Returns empty string if no answer found in \boxed.
        """
        try:
            answer = remove_boxed(last_boxed_only_string(output))
            return answer
        except:
            return ""
