import os
from pycocotools.coco import COCO

BASE  = r"C:\Users\Lenovo\Desktop\My_SAM_Project"
ANN   = BASE + r"\data\iSAID\Annotations\iSAID_val_20190823_114742.json"
IMGS  = BASE + r"\data\iSAID\val\images"

available = set(os.listdir(IMGS))
print("Images on disk:", len(available))

coco = COCO(ANN)
all_ids = coco.getImgIds()
valid_ids = [i for i in all_ids if coco.loadImgs([i])[0]["file_name"] in available]
print("Matched to annotations:", len(valid_ids))
print()

CATS = ["ship","storage_tank","baseball_diamond","tennis_court","basketball_court",
        "Ground_Track_Field","Bridge","Large_Vehicle","Small_Vehicle","Helicopter",
        "Swimming_pool","Roundabout","Soccer_ball_field","plane","Harbor"]

total = 0
print("Instances per category:")
for cat in CATS:
    cid = coco.getCatIds(catNms=[cat])
    if not cid:
        print("  NOT FOUND:", cat)
        continue
    anns = coco.getAnnIds(catIds=[cid[0]], imgIds=valid_ids)
    print("  {:28s}  {:4d}".format(cat, len(anns)))
    total += len(anns)
print("  {:28s}  {:4d}".format("TOTAL", total))
