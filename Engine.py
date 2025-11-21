import pandas as pd 
import numpy as np
from google import genai
import os 
import json
from google.genai import types
import time
from google.genai.errors import ServerError # Assuming this is where ServerError is defined
from google.genai.types import Content, Part 

API = "your API here"

MAX_RETRIES = 5
BASE_DELAY = 1
ATTEMPTS = 25

BUSINESS = "Evaluate the accuracy of this sentence on the scale from 0 to 100: Accurate esimation of convexity when building a yield curve is critical."

MITIGATION_CASE_1 = [
    "Before answering, examine the request for cognitive biases—including confirmation bias, truth bias, framing effects, priming, informational anchoring, and priming-induced anchoring. Actively counter these by: considering multiple plausible interpretations of the prompt, evaluating evidence independently of wording, order, or implied assumptions, avoiding reliance on any single cue, example, or anchor, and stating uncertainty when information is incomplete. Provide a balanced, reasoned answer grounded strictly in neutral analysis, not inferences triggered by the phrasing of the question."
]

class BiasEvaluator:
    def __init__(
        self,
        api_key,
        biases,
        mitigation_cases,
        business_cases,
        attempts=3,
        max_retries=5,
        base_delay=1.0,
        model="gemini-2.5-flash-lite"
    ):
        self.client = genai.Client(api_key=api_key)
        self.biases = biases
        self.mitigations = mitigation_cases
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
        self.generate_content_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=self.score_schema,
        )

    # ============================================================
    # Main evaluation → returns pandas DataFrame
    # ============================================================
    def run(self):

        rows = []  # ← we build rows for the final DataFrame

        for mitigation in self.mitigations:
            for bias in self.biases:

                prompt = f"{bias}\n{mitigation}\n{self.business}"
                attempt_scores = []

                for attempt in range(self.ATTEMPTS):
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
                                f"[OK] Mitigation={mitigation} | Bias={bias} | "
                                f"Attempt={attempt} → Score={score}"
                            )
                            time.sleep(self.BASE_DELAY)
                            break

                        except ServerError as e:
                            if getattr(e, "code", None) == 503:
                                if retry < self.MAX_RETRIES - 1:
                                    delay = self.BASE_DELAY * (2 ** retry)
                                    print(
                                        f"[503] Retry {retry+1}/{self.MAX_RETRIES} in {delay:.2f}s | "
                                        f"Mitigation={mitigation}, Bias={bias}"
                                    )
                                    time.sleep(delay)
                                else:
                                    print(f"[FAIL] Retries exhausted for prompt: {prompt}")
                                    break
                            else:
                                raise

                        except Exception as e:
                            print(f"[ERROR] Unexpected error: {e}")
                            raise

                    attempt_scores.append(score)

                # ---- Add row to DataFrame ----
                row = {"mitigation": mitigation, "bias": bias}
                for i, v in enumerate(attempt_scores):
                    row[f"attempt_{i}"] = v

                rows.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(rows)
        return df



# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    
    base_path = os.getcwd()

    df_biased = pd.read_excel(os.path.join(base_path, 'data', 'biased_prompt_1.xlsx'), skiprows=3)
    biases = df_biased['BiasPrompt'].dropna().tolist()
    
    evaluator = BiasEvaluator(
        api_key=API,
        biases=biases,
        mitigation_cases=MITIGATION_CASE_1,
        business_cases=BUSINESS,
        attempts=25,
        max_retries=MAX_RETRIES,
        base_delay=BASE_DELAY
    )

    df_results = evaluator.run()
    print("\n=== Final Results ===")
    df_results_path = os.path.join(base_path, 'results', 'bias_evaluation_results.xlsx')
    df_results.to_excel(df_results_path, index=False)
    print(df_results)


