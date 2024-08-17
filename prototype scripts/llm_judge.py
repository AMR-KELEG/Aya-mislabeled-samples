import cohere 
import re
import pandas as pd

co = cohere.Client(
  api_key="", # This is your API key
) 

def eval(row):
  inputprompt = row["inputs"]
  answer = row["targets"]

  print(inputprompt, answer)

  response = co.chat( 
    model='c4ai-aya-23-8b',
    message=f'''[System]
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user question displayed below. Your evaluation should consider factors
such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of
the response. Begin your evaluation by providing a short explanation. Be as objective as
possible. Direct translations of the input prompt should NOT be rated highly. After providing your explanation, please rate the response on a scale of 1 to 10
by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".
  [Input Prompt]
  {inputprompt}
  [The Start of Assistant’s Answer]
  {answer}
  [The End of Assistant’s Answer]''',
  ) 
  return response.text

def extract_rating(s):
    match = re.search(r'\[(\d+(\.\d+)?)\]', s)
    if match:
        return float(match.group(1))
    return None

df = pd.read_csv("hf://datasets/mislabel-indentification-aya/aya_dataset_mislabeled/mislabeled_samples_CohereForAI_aya_dataset_train.csv")

df["llm_judge"] = df.apply(eval, axis=1)
df["rating"] = df["llm_judge"].apply(extract_rating)

df.to_csv("aya_mislabelled_with_evaluation.csv")
