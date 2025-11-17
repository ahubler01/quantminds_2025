import pandas as pd 
import numpy as np
from google import genai
import os 
import json
from google.genai import types
import time
from google.genai.errors import ServerError # Assuming this is where ServerError is defined
from google.genai.types import Content, Part 
import random
import optuna
from tqdm import tqdm

# API='AIzaSyBNsdENpvh4cTNdgGBQ_zKk09UlYzHsbvo'
API="AIzaSyCF0Ab8hjQflL5CW0-RR1t6PNTCXXuvT_A"

MAX_RETRIES = 5
BASE_DELAY = 1
ATTEMPTS = 25

BUSINESS = "Evaluate the accuracy of this sentence on the scale from 0 to 100: Accurate esimation of convexity when building a yield curve is critical."



class BiasEvaluator:
    def __init__(
        self,
        api_key,
        biases_df,              # <- DataFrame with BiasPrompt, ExpectedValue
        mitigation_cases,       # list of mitigation strings
        business_cases,         # whatever you had before
        attempts=3,
        max_retries=5,
        base_delay=1.0,
        model="gemini-2.5-flash-lite",
    ):
        self.client = genai.Client(api_key=api_key)
        self.biases = biases_df                 # DataFrame with columns: BiasPrompt, ExpectedValue
        self.mitigations = mitigation_cases     # full pool of mitigations
        self.business = business_cases
        self.ATTEMPTS = attempts
        self.MAX_RETRIES = max_retries
        self.BASE_DELAY = base_delay
        self.model = model

        # ---- Schema ----
        self.score_schema = genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["score"],
            properties={
                "score": genai.types.Schema(type=genai.types.Type.NUMBER),
            },
        )

        # ---- Output config ----
        self.generate_content_config = genai.types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=self.score_schema,
        )

    # ============================================================
    # Evaluate a *subset* of mitigations with pruning support
    # ============================================================
    def evaluate_subset(self, mitigation_subset, trial=None, prune_start_step=0):
        """
        mitigation_subset: list of mitigations to evaluate.
        trial: Optuna trial (can be None if you just want to run it).
        prune_start_step: after which attempt index pruning is allowed.

        For each combination (mitigation, bias) we will do ATTEMPTS calls.
        The loop structure is:

            for attempt in [0..ATTEMPTS-1]:
                for each bias in biases_df:
                    for each mitigation in mitigation_subset:
                        call model once

        After each full pass over all biases (i.e. after a given attempt),
        we compute a running mean score and report it to Optuna.
        """

        # (mitigation, bias_idx) -> list of scores (one per attempt)
        scores_dict = {}

        # Pre-initialise containers
        for mitigation in mitigation_subset:
            for bias_idx in self.biases.index:
                scores_dict[(mitigation, bias_idx)] = []

        # ===== main evaluation loop =====
        for attempt in range(self.ATTEMPTS):

            # For this attempt, iterate over all biases
            for bias_idx, bias_row in self.biases.iterrows():
                bias_prompt = bias_row["BiasPrompt"]
                expected_value = bias_row["ExpectedValue"]  # kept in DF for analysis if needed

                for mitigation in mitigation_subset:
                    prompt = f"{bias_prompt}\n{mitigation}\n{self.business}"

                    score = None
                    for retry in range(self.MAX_RETRIES):
                        try:
                            response = self.client.models.generate_content(
                                model=self.model,
                                contents=[
                                    Content(
                                        role="user",
                                        parts=[Part.from_text(text=prompt)],
                                    )
                                ],
                                config=self.generate_content_config,
                            )

                            score = response.parsed["score"]
                            print(
                                f"[OK] Mitigation={mitigation[:40]}... | "
                                f"Bias={bias_prompt[:40]}... | Attempt={attempt} → Score={score}"
                            )
                            time.sleep(self.BASE_DELAY)
                            break

                        except ServerError as e:
                            if getattr(e, "code", None) == 503:
                                if retry < self.MAX_RETRIES - 1:
                                    delay = self.BASE_DELAY * (2 ** retry)
                                    print(
                                        f"[503] Retry {retry+1}/{self.MAX_RETRIES} in {delay:.2f}s | "
                                        f"Mitigation={mitigation[:30]}..., Bias={bias_prompt[:30]}..."
                                    )
                                    time.sleep(delay)
                                else:
                                    print(f"[FAIL] Retries exhausted for prompt: {prompt[:80]}...")
                                    break
                            else:
                                raise

                        except Exception as e:
                            print(f"[ERROR] Unexpected error: {e}")
                            raise

                    scores_dict[(mitigation, bias_idx)].append(score)

            # --- after finishing all biases for this attempt → report to pruner ---
            all_scores_so_far = [
                s for lst in scores_dict.values() for s in lst if s is not None
            ]
            if all_scores_so_far:
                running_mean = sum(all_scores_so_far) / len(all_scores_so_far)

                if trial is not None:
                    trial.report(running_mean, step=attempt)
                    if attempt >= prune_start_step and trial.should_prune():
                        # Optuna will handle this as a pruned trial
                        raise optuna.TrialPruned()

        # ===== Build the output DataFrame =====
        rows = []
        for (mitigation, bias_idx), scores in scores_dict.items():
            bias_row = self.biases.loc[bias_idx]
            row = {
                "mitigation": mitigation,
                "BiasPrompt": bias_row["BiasPrompt"],
                "ExpectedValue": bias_row["ExpectedValue"],
            }
            for i, v in enumerate(scores):
                row[f"attempt_{i}"] = v
            rows.append(row)

        df = pd.DataFrame(rows)

        # final objective: mean over all attempt_* columns
        attempt_cols = [c for c in df.columns if c.startswith("attempt_")]
        final_score = df[attempt_cols].to_numpy().mean()

        return df, final_score
    

def optimize_mitigations(
    evaluator: BiasEvaluator,
    n_trials: int = 20,
    initial_group_size: int = 5,
    chunk_size: int = 3,
    prune_start_step: int = 1,     # when inside a trial (attempt index) pruning may start
    n_startup_trials: int = 3,     # how many trials before pruner activates
):
    """
    evaluator: BiasEvaluator instance.
    n_trials: total Optuna trials.
    initial_group_size: size of the first random mitigation group (baseline).
    chunk_size: how many new mitigations to add each trial.
    prune_start_step: attempt index from which to allow pruning inside a trial.
    n_startup_trials: how many trials before the pruner starts pruning.

    Returns:
        study, best_df
    """

    pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials)
    study = optuna.create_study(direction="maximize", pruner=pruner)

    mitigations_pool = list(evaluator.mitigations)
    best_df_container = {"df": None}

    def objective(trial: optuna.trial.Trial):

        # === Determine which mitigation subset to evaluate in this trial ===
        if trial.number == 0 or "best_subset" not in study.user_attrs:
            # First trial -> random baseline group
            current_subset = set(
                random.sample(
                    mitigations_pool,
                    min(initial_group_size, len(mitigations_pool))
                )
            )
        else:
            best_subset = set(study.user_attrs["best_subset"])
            remaining = list(set(mitigations_pool) - best_subset)

            if remaining:
                # Add a new random "chunk" of mitigations
                k = min(chunk_size, len(remaining))
                new_chunk = set(random.sample(remaining, k))
                current_subset = best_subset | new_chunk
                trial.set_user_attr("new_chunk", list(new_chunk))
            else:
                # Nothing left to add – just evaluate the current best subset again
                current_subset = best_subset

        current_subset_list = list(current_subset)

        # === Evaluate this subset with the evaluator, including pruning ===
        df, score = evaluator.evaluate_subset(
            mitigation_subset=current_subset_list,
            trial=trial,
            prune_start_step=prune_start_step,
        )

        # === If we improved, update best subset and keep the DataFrame ===
        if "best_score" not in study.user_attrs or score > study.user_attrs["best_score"]:
            study.set_user_attr("best_score", float(score))
            study.set_user_attr("best_subset", current_subset_list)
            best_df_container["df"] = df

        return score

    study.optimize(objective, n_trials=n_trials)

    best_df = best_df_container["df"]
    return study, best_df
    
    
# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    
    base_path = os.getcwd()

    df_biased = pd.read_excel(os.path.join(base_path, 'data', 'Biased Prompt 1.xlsx'), skiprows=3).dropna(subset=['BiasPrompt'])
    df_biased = df_biased.sort_values('ExpectedValue')
    df_biased = df_biased.drop_duplicates(subset='ExpectedValue', keep='first')
    
    df_mitigation = pd.read_csv(os.path.join(base_path, 'data', 'bricks.csv'))
    mitigations = df_mitigation['brick_text'].dropna().tolist()
    
    evaluator = BiasEvaluator(
        api_key=API,
        biases_df=df_biased,
        mitigation_cases=mitigations,
        business_cases=BUSINESS,
        attempts=5,           # how many "attempt cycles" per combination
        max_retries=5,
        base_delay=1.0,
        model="gemini-2.5-flash-lite",
    )

    study, best_df = optimize_mitigations(
        evaluator=evaluator,
        n_trials=4,
        initial_group_size=5,
        chunk_size=3,
        prune_start_step=1,      # start pruning after attempt 1 (inside each trial)
        n_startup_trials=3,      # first 3 trials always run fully
    )

    print("Best value:", study.best_value)
    print("Best mitigation subset:", study.user_attrs["best_subset"])
    print(best_df.head())

    df_results_path = os.path.join(base_path, 'results', 'optimised_bricks.xlsx')
    best_df.to_excel(df_results_path, index=False)


