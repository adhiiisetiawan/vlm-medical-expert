# from datasets import load_dataset
# ds = load_dataset("flaviagiammarino/path-vqa")

# import pickle
# with open("test_pathvqa.pkl", "wb") as f:
#     pickle.dump(ds["test"], f)
import pickle
import json

# with open('/mnt/extended-home/adhi/dataset/path_vqa/pvqa/qas/test_vqa.pkl', 'rb') as f:
#     obj = pickle.load(f)
#     print(type(obj))
#     print(obj[:100])
#     json.dump(obj, open('test_vqa.json', 'w'))


with open('output.pkl', 'rb') as f:
    obj = pickle.load(f)
    print(type(obj))
    print(obj[:100])
    json.dump(obj, open('output.json', 'w'))
