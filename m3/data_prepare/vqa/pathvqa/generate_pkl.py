import pandas as pd
import pickle

# Load your CSV
df = pd.read_csv("/mnt/extended-home/adhi/dataset/path_vqa/data/test_metadata.csv")

# Convert to required pkl format
pkl_data = []

for _, row in df.iterrows():
    answer = str(row["answer"]).strip().lower()
    
    # Determine answer_type
    if answer in ["yes", "no"]:
        answer_type = "yes/no"
    else:
        answer_type = "other"

    entry = {
        "question_id": row["file_name"],  # include .jpg as is
        "answer_type": answer_type,
        "label": {answer: 1.0}
    }
    pkl_data.append(entry)

# Save as pickle
with open("output.pkl", "wb") as f:
    pickle.dump(pkl_data, f)

print(f"Saved {len(pkl_data)} entries to output.pkl")
