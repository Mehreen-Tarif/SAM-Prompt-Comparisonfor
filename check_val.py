import os, json
from pycocotools.coco import COCO

BASE = r"C:\Users\Lenovo\Desktop\My_SAM_Project"
ANN  = BASE + r"\data\iSAID\Annotations\iSAID_val.json"

with open(ANN) as f:
    d = json.load(f)

print("Keys in file:", list(d.keys()))
print("Images:", len(d.get("images", [])))
print("Annotations:", len(d.get("annotations", [])))
print("Categories:", len(d.get("categories", [])))

if d.get("annotations"):
    a = d["annotations"][0]
    print()
    print("Sample annotation keys:", list(a.keys()))
    print("Has segmentation?", "segmentation" in a)
    print("Has bbox?", "bbox" in a)
    if "segmentation" in a:
        print("Segmentation type:", type(a["segmentation"]))
        print("Segmentation sample:", str(a["segmentation"])[:100])
