import requests
import json
import textwrap
from collections import Counter
import pandas as pd
from typing import Optional, Tuple, List

class LLMDetector():
    """
    Detector that uses Ollama to classify fraudulent and legit 
    FAST-payment sequences.
    """

    def __init__(self, coev_file_path, model):
        self.coev_file_path = coev_file_path
        self.model = model
    
    def classify_sequence(self, seq: str, max_attempts: int = 5, timeout_s: int = 60) -> Optional[str]:
        """
        Classifies one financial sequence in string format
        Given 5 attempts to generate single deterministic output "fraud" or "legit"

        Args:
            seq (str): financial sequence
        Returns:
            res: "fraud", "legit", or None if sequence was unclassifiable
        """

        prompt = textwrap.dedent(f"""\
        You are a strict binary classifier for FAST-payment sequences.

        Your job: label the *behavior* described in the sequence.
        - "fraud" = unauthorized / deceptive activity leading to an improper transfer (phishing, impersonation, credential theft, account takeover, SIM swap, coercion, etc.)
        - "legit" = ordinary, authorized payments for goods/services, bills, tuition, rent, reimbursements, etc.

        IMPORTANT:
        - Do NOT treat this as a "system log" task.
        - The presence of the words "phishing", "fraudulent", "malware", etc. indicates FRAUD behavior (not "legit input").

        Output rules (MUST follow):
        - Output exactly ONE word: fraud OR legit
        - No punctuation, no quotes, no extra words, no explanations.

        Input sequence:
        {seq}

        """).strip() + "\n"

        attempts = 0

        while attempts < max_attempts:
            attempts += 1
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0,
                            # keep generations short to discourage rambling
                            "num_predict": 4,
                            # stop as soon as it tries to add anything beyond the first word
                            "stop": ["\n"],
                        },
                    },
                    timeout=timeout_s,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception:
                # network/timeout/HTTP/JSON issues -> treat as failed attempt
                continue

            raw = (payload.get("response") or "").strip()
            # If the model rambles, keep only the first line to reduce drift
            raw = raw.splitlines()[0].strip() if raw else ""

            # normalize to one token
            token = raw.split()[0] if raw.split() else ""
            token = token.strip().strip('"\'').strip(".,:;!?()[]{}")
            token = token.lower()

            # allow common near-misses
            if token in {"fraud", "fraudulent"} or token.startswith("fraud"):
                return "fraud"
            if token in {"legit", "legitimate"} or token.startswith("legit"):
                return "legit"

            # If we got an invalid output, append a minimal corrective instruction and retry
            prompt += (
                "\nYour previous output was invalid: " + raw + "\n"
                "Remember: output exactly one word: fraud OR legit.\nOutput:\n"
            )
            
        return None
    

    def ensemble_classify_sequence(self, seq: str, num_calls: int = 5) -> Tuple[Optional[str], List[Optional[str]], float, float]:
        """
        Runs classify_sequence on financial sequence num_calls times, gets "winner"
        Assess agreeableness between detection runs and whether model is stable
        """
        labels: List[Optional[str]] = [self.classify_sequence(seq) for _ in range(num_calls)]
        valid = [x for x in labels if x in ("fraud", "legit")]

        valid_rate = len(valid) / num_calls if num_calls else 0.0
        if not valid:
            return None, labels, 0.0, valid_rate

        winner, count = Counter(valid).most_common(1)[0]
        # stability among valid votes (agreement)
        stability = count / len(valid)
        return winner, labels, stability, valid_rate


    def explain_classification(self, seq: str, result: str) -> str:
        """
        Prompts LLM to explain reasoning for classification
        Args:
            seq: financial sequence
            result: classification from ensemble_classify_sequence
        Returns:
            reason: explanation for classification
        """

        prompt = (
            f"Explain why you classified this sequence as {result}.\n\n"
            f"Input:\n{seq}\n\nOutput:"
        )

        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': self.model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    "temperature": 0
                }
            }
        )
        res = response.json().get('response', '').strip().strip("'")
        return res
    
    def run_detector(self):
        """
        Run detector on all sequences
        """
        error_seq = []
        res = []

        num_correct = 0
        false_pos = 0
        false_neg = 0
        total_seq = 0
        unclassifiable = 0

        with open(self.coev_file_path, "r") as f:
            data = json.load(f)['sequences']

            for sequence in data:
                total_seq += 1
                print(f"Classifying Sequence {sequence['sequence_id']}.")

                classification, labels, stability, valid_rate = self.ensemble_classify_sequence(sequence)

                if classification == "fraud":
                    num_correct += 1
                print(classification)

                # if classification == label:
                #     num_correct += 1
                # else:
                #     if classification is None:
                #         unclassifiable += 1
                #     elif classification == "fraud" and label == "legit":
                #         false_pos += 1
                #     elif classification == "legit" and label == "fraud":
                #         false_neg += 1
                #     else:
                #         unclassifiable += 1
                #     error_seq.append({
                #         'Sequence id': id,
                #         'Sequence': sequence,
                #         'Label': label,
                #         'LLM Generated Label': classification,
                #         'Stability': stability,
                #         'Valid rate': valid_rate,
                #         'Votes': labels,
                #     })
            
            # df_error = pd.DataFrame(error_seq)
            # print(df_error)
            # df_error.to_csv("data/detector/v2/detector_errors_v2_5.csv", index=False)

        # res.append({
        #     'Accuracy': num_correct/total_seq,
        #     'False positive': false_pos/total_seq,
        #     'False negative': false_neg/total_seq,
        #     'Unclassifiable': unclassifiable/total_seq
        # })
        # df_res = pd.DataFrame(res)
        # print(df_res)
        # df_res.to_csv("data/detector/v2/detector_res_v2_5.csv", index=False)
        # return num_correct/total_seq, false_pos/total_seq, false_neg/total_seq, unclassifiable/total_seq
        return num_correct/total_seq


def main():
    detector = LLMDetector("sequences_100.json", "llama3.2")
    accuracy = detector.run_detector()
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()


# ## Model evaluation for detector

# models = ['llama3.2', 'chevalblanc/gpt-4o-mini', 'mistral','gemma3:4b']
# model_res = []

# for model in models:
#     print(f"Testing model {model} -----------------------------------------")
#     detector = LLMDetector("data/coev/coev_seq_v2.json", model)
#     acc, false_p, false_n, unclass = detector.run_detector()
#     model_res.append({
#         'Model': model,
#         'Accuracy': acc,
#         'False positive': false_p,
#         'False negative': false_n,
#         'Unclassifiable': unclass
#     })
