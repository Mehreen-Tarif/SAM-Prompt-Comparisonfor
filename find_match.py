import os
from pycocotools.coco import COCO

BASE  = r"C:\Users\Lenovo\Desktop\My_SAM_Project"
IMGS  = BASE + r"\data\iSAID\val\images"
available = set(os.listdir(IMGS))
print("Images on disk:", sorted(list(available))[:10])
print()

ann_files = [
    BASE + r"\data\iSAID\Annotations\iSAID_val.json",
    BASE + r"\data\iSAID\Annotations\iSAID_train.json",
    BASE + r"\data\iSAID\Annotations\iSAID_val_20190823_114742.json",
    BASE + r"\data\iSAID\Annotations\iSAID_train_20190823_114751.json",
]

for ann_path in ann_files:
    if not os.path.exists(ann_path):
        continue
    fname = os.path.basename(ann_path)
    coco = COCO(ann_path)
    all_ids = coco.getImgIds()
    matched = [i for i in all_ids if coco.loadImgs([i])[0]["file_name"] in available]
    ann_names = set(coco.loadImgs([i])[0]["file_name"] for i in all_ids[:20])
    print("File:", fname)
    print("  Total images in annotation:", len(all_ids))
    print("  Matched to your disk:", len(matched))
    if matched:
        print("  Sample matched:", [coco.loadImgs([i])[0]["file_name"] for i in matched[:5]])
    print()
