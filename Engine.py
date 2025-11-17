import pandas as pd 
import numpy as np
from google import genai
import os 

from google.genai import types
import time
from google.genai.errors import ServerError # Assuming this is where ServerError is defined
from google.genai.types import Content, Part 

API='AIzaSyBNsdENpvh4cTNdgGBQ_zKk09UlYzHsbvo'
MAX_RETRIES = 5
BASE_DELAY = 1
ATTEMPTS = 25

base_path = os.getcwd()

df = pd.read_excel(os.path.join(base_path, 'data', 'biased_prompt_1.xlsx'))

client = genai.Client(api_key=API)

score_schema = genai.types.Schema(
    type=genai.types.Type.OBJECT,
    required=["score"],
    properties={
        "score": genai.types.Schema(
            type=genai.types.Type.NUMBER,
        ),
    },
)

generate_content_config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=score_schema,
)

results = {}
for bias in biases:
    results[bias] = {}

    for mitigation in mitigation_cases:
        prompt = f"{bias}\n{mitigation}\n{business[0]}"

        # Ensure a dict exists for this mitigation before writing attempts into it
        results[bias][mitigation] = {}
        
        for attempt in range(ATTEMPTS):
            for retry in range(MAX_RETRIES):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[
                            Content(
                                role="user",
                                parts=[Part.from_text(text=prompt)],
                            ),
                        ],
                        config=generate_content_config,
                    )
                    
                    score = response.parsed["score"]
                    results[bias][mitigation][attempt] = score
                    print(f"{bias} -> {score}")
                    
                    time.sleep(BASE_DELAY)
                    break
                    
                except ServerError as e:
                    # Check for 503 error using the .code attribute, which typically holds the status code
                    if hasattr(e, 'code') and e.code == 503:
                        if retry < MAX_RETRIES - 1:
                            delay = BASE_DELAY * (2 ** retry)
                            print(f"ServerError 503 (attempt {retry + 1}/{MAX_RETRIES}). Retrying in {delay:.2f} seconds...")
                            time.sleep(delay)
                        else:
                            print(f"ServerError 503: Failed after {MAX_RETRIES} attempts for prompt: {prompt}")
                            break
                    else:
                        print(f"Non-503 ServerError or other error: {e}")
                        raise
                except Exception as e:
                    print(f"An unexpected error occurred for prompt: {prompt}. Error: {e}")
                    raise








