import pandas as pd
import json

# text normal
def norm(text):
    return "".join(str(text).lower().split())

# compute Hit@k
def hit_at_k(docs, gt):
    gt_norm = norm(gt)
    return any(gt_norm in norm(d) for d in docs)

# main eval rag 
def evaluate_rag(eval_df):
    hr = []
    for _, row in eval_df.iterrows():
        docs = json.loads(row['retrieved_docs'])
        gt = row['answer_true']
        hr.append(hit_at_k(docs, gt))
    metrics = sum(hr) / len(hr)
    return metrics

df = pd.read_csv("eval_data.csv")
metrics = evaluate_rag(df)
print(f"Hit@k: {metrics:.4f}")

