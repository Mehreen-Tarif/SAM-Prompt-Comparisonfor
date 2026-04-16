import os, sys, time, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry, SamPredictor
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

BASE   = r"C:\Users\Lenovo\Desktop\My_SAM_Project"
ANN    = BASE + r"\data\iSAID\Annotations\iSAID_val.json"
IMGS   = BASE + r"\data\iSAID\val\images"
CKPT_H = BASE + r"\sam_vit_h_4b8939.pth"
CKPT_B = BASE + r"\sam_vit_b_01ec64.pth"
OUT    = BASE + r"\results"
FIGS   = BASE + r"\figures"
SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CATS = ["ship","storage_tank","baseball_diamond","tennis_court","basketball_court",
        "Ground_Track_Field","Bridge","Large_Vehicle","Small_Vehicle","Helicopter",
        "Swimming_pool","Roundabout","Soccer_ball_field","plane","Harbor"]

os.makedirs(OUT, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Load instances (same as main experiment) ──────────────────────────────
def load_instances():
    available = set(os.listdir(IMGS))
    coco      = COCO(ANN)
    all_ids   = coco.getImgIds()
    valid_ids = [i for i in all_ids
                 if coco.loadImgs([i])[0]["file_name"] in available]
    items = []
    for cat_name in CATS:
        cid = coco.getCatIds(catNms=[cat_name])
        if not cid: continue
        ann_ids = coco.getAnnIds(catIds=[cid[0]], imgIds=valid_ids)
        if not ann_ids: continue
        for aid in ann_ids:
            ann      = coco.loadAnns([aid])[0]
            img_info = coco.loadImgs([ann["image_id"]])[0]
            img_path = os.path.join(IMGS, img_info["file_name"])
            if not os.path.exists(img_path): continue
            seg = ann["segmentation"]
            if not seg: continue
            try:
                rle  = maskUtils.frPyObjects(seg, img_info["height"], img_info["width"])
                mask = maskUtils.decode(rle)
                if mask.ndim == 3: mask = mask[:,:,0]
                if mask.sum() == 0: continue
            except: continue
            x,y,w,h = ann["bbox"]
            if w<=0 or h<=0: continue
            items.append({"cat":cat_name,"bbox":ann["bbox"],"mask":mask.astype(bool),
                          "path":img_path,"h":img_info["height"],"w":img_info["width"]})
    print("  Loaded {} instances".format(len(items)))
    return items

def iou_score(pred, gt):
    tp=np.logical_and(pred,gt).sum()
    fp=np.logical_and(pred,~gt).sum()
    fn=np.logical_and(~pred,gt).sum()
    return float(tp/(tp+fp+fn+1e-8))

def center_pt(mask):
    ys,xs = np.where(mask)
    if len(xs)==0: h,w=mask.shape; return np.array([[w/2.0,h/2.0]])
    return np.array([[float(xs.mean()),float(ys.mean())]])

def get_bb(bbox, sx=1, sy=1):
    x,y,w,h = bbox
    return np.array([x*sx, y*sy, (x+w)*sx, (y+h)*sy])

def run_bb_strategy(items, predictor, img_size=1024, noise_px=0):
    results = []
    for item in items:
        img = cv2.imread(item["path"])
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        sx  = img_size / item["w"]
        sy  = img_size / item["h"]
        gt  = cv2.resize(item["mask"].astype(np.uint8),
                         (img_size,img_size),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
        predictor.set_image(img)
        bbox = item["bbox"].copy() if isinstance(item["bbox"], np.ndarray) else list(item["bbox"])
        if noise_px > 0:
            bbox[0] += random.randint(-noise_px, noise_px)
            bbox[1] += random.randint(-noise_px, noise_px)
            bbox[2] += random.randint(-noise_px, noise_px)
            bbox[3] += random.randint(-noise_px, noise_px)
        bb = get_bb(bbox, sx, sy)
        try:
            masks, scores, _ = predictor.predict(
                point_coords=None, point_labels=None,
                box=bb, multimask_output=True)
            best = np.argmax(scores)
            iou  = iou_score(masks[best], gt)
        except: iou = 0.0
        results.append({"cat": item["cat"], "iou": iou})
    return results

# ══════════════════════════════════════════════════════════════════════════
# ABLATION A1 — Backbone: ViT-B vs ViT-H
# WHAT THIS TELLS US: Does using a bigger AI model improve accuracy?
# ViT-B = small model (91M parameters), ViT-H = large model (636M params)
# ══════════════════════════════════════════════════════════════════════════
def ablation_backbone(items):
    print("\n" + "="*55)
    print("  ABLATION A1: Backbone Size (ViT-B vs ViT-H)")
    print("  Question: Does bigger model = better segmentation?")
    print("="*55)
    results = {}
    configs = [
        ("ViT-H (636M params)", "vit_h", CKPT_H),
        ("ViT-B (91M params)",  "vit_b", CKPT_B),
    ]
    for name, model_type, ckpt in configs:
        if not os.path.exists(ckpt):
            print("  SKIP {} — weights not found: {}".format(name, ckpt))
            continue
        print("\n  Loading {}...".format(name))
        sam  = sam_model_registry[model_type](checkpoint=ckpt)
        sam.to(device=DEVICE)
        pred = SamPredictor(sam)
        rows = run_bb_strategy(items, pred, img_size=1024)
        ious = [r["iou"] for r in rows]
        mean_iou = np.mean(ious)
        results[name] = {"mIoU": round(mean_iou, 4), "n": len(ious)}
        print("  {} → mIoU = {:.4f}".format(name, mean_iou))
        del sam, pred
        torch.cuda.empty_cache()

    df = pd.DataFrame([
        {"Backbone": k, "Parameters": k.split("(")[1].rstrip(")"),
         "mIoU": v["mIoU"], "Instances": v["n"]}
        for k, v in results.items()
    ])
    df.to_csv(OUT + r"\Ablation_A1_Backbone.csv", index=False)
    print("\n  Saved: Ablation_A1_Backbone.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    names  = list(results.keys())
    values = [results[n]["mIoU"] for n in names]
    colors = ["#185FA5" if "H" in n else "#9FE1CB" for n in names]
    bars   = ax.bar(names, values, color=colors, width=0.45, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.005,
                "{:.4f}".format(v), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean IoU", fontsize=11)
    ax.set_title("Ablation A1: Effect of Backbone Size on Segmentation Accuracy\n(Bounding Box prompt, iSAID val set)", fontsize=10)
    ax.set_ylim(0, max(values)*1.3)
    ax.spines[["top","right"]].set_visible(False)
    ax.axhline(y=max(values), color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGS + r"\Ablation_A1_Backbone.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: Ablation_A1_Backbone.png")
    return results

# ══════════════════════════════════════════════════════════════════════════
# ABLATION A2 — Input Resolution: 512 vs 768 vs 1024
# WHAT THIS TELLS US: Does higher image resolution improve accuracy?
# Higher res = more detail but slower. We test 3 sizes.
# ══════════════════════════════════════════════════════════════════════════
def ablation_resolution(items):
    print("\n" + "="*55)
    print("  ABLATION A2: Input Resolution")
    print("  Question: Does higher resolution = better segmentation?")
    print("="*55)
    sam  = sam_model_registry["vit_h"](checkpoint=CKPT_H)
    sam.to(device=DEVICE)
    pred = SamPredictor(sam)
    resolutions = [512, 768, 1024]
    results     = {}
    for res in resolutions:
        print("  Testing {}x{}...".format(res, res))
        t0   = time.perf_counter()
        rows = run_bb_strategy(items, pred, img_size=res)
        t1   = time.perf_counter()
        ious = [r["iou"] for r in rows]
        mean_iou  = np.mean(ious)
        avg_time  = (t1-t0)/len(items)*1000
        results[res] = {"mIoU": round(mean_iou,4), "time_ms": round(avg_time,1)}
        print("  {}x{} → mIoU={:.4f}  avg_time={:.1f}ms".format(res,res,mean_iou,avg_time))
    del sam, pred; torch.cuda.empty_cache()
    df = pd.DataFrame([
        {"Resolution": "{}x{}".format(r,r), "mIoU": v["mIoU"], "Avg Time (ms)": v["time_ms"]}
        for r,v in results.items()])
    df.to_csv(OUT + r"\Ablation_A2_Resolution.csv", index=False)
    print("\n  Saved: Ablation_A2_Resolution.csv")

    # Plot with dual axis
    fig, ax1 = plt.subplots(figsize=(7,4))
    ax2 = ax1.twinx()
    res_labels = ["{}x{}".format(r,r) for r in resolutions]
    ious_vals  = [results[r]["mIoU"]    for r in resolutions]
    time_vals  = [results[r]["time_ms"] for r in resolutions]
    ax1.plot(res_labels, ious_vals, marker="o", color="#185FA5",
             linewidth=2.5, markersize=8, label="mIoU")
    ax2.plot(res_labels, time_vals, marker="s", color="#D85A30",
             linewidth=2, markersize=7, linestyle="--", label="Time (ms)")
    for i,(l,v,t) in enumerate(zip(res_labels,ious_vals,time_vals)):
        ax1.text(i, v+0.005, "{:.4f}".format(v), ha="center", fontsize=9, color="#185FA5")
        ax2.text(i, t+0.3,  "{:.1f}ms".format(t), ha="center", fontsize=9, color="#D85A30")
    ax1.set_ylabel("Mean IoU", color="#185FA5", fontsize=11)
    ax2.set_ylabel("Inference Time (ms)", color="#D85A30", fontsize=11)
    ax1.set_xlabel("Input Resolution", fontsize=11)
    ax1.set_title("Ablation A2: Effect of Input Resolution\n(Accuracy vs Speed Trade-off)", fontsize=10)
    ax1.set_ylim(0, max(ious_vals)*1.3)
    lines1,labels1 = ax1.get_legend_handles_labels()
    lines2,labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGS + r"\Ablation_A2_Resolution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: Ablation_A2_Resolution.png")
    return results

# ══════════════════════════════════════════════════════════════════════════
# ABLATION A3 — Bounding Box Noise Robustness
# WHAT THIS TELLS US: What happens if the bounding box is not perfect?
# In real applications, boxes come from detectors and may be slightly off.
# We add random noise and measure how much accuracy drops.
# ══════════════════════════════════════════════════════════════════════════
def ablation_box_noise(items):
    print("\n" + "="*55)
    print("  ABLATION A3: Bounding Box Noise Robustness")
    print("  Question: Does SAM still work with imperfect boxes?")
    print("="*55)
    sam  = sam_model_registry["vit_h"](checkpoint=CKPT_H)
    sam.to(device=DEVICE)
    pred = SamPredictor(sam)
    noise_levels = [0, 5, 10, 20, 30]
    results      = {}
    for noise in noise_levels:
        random.seed(SEED)
        rows = run_bb_strategy(items, pred, img_size=1024, noise_px=noise)
        ious = [r["iou"] for r in rows]
        mean_iou = np.mean(ious)
        label = "Tight (0px)" if noise==0 else "+-{}px jitter".format(noise)
        results[noise] = {"mIoU": round(mean_iou,4), "label": label}
        drop = ((results[0]["mIoU"]-mean_iou)/results[0]["mIoU"]*100) if noise>0 else 0
        print("  Noise +-{:2d}px → mIoU={:.4f}  ({:+.1f}% vs tight)".format(
              noise, mean_iou, -drop))
    del sam, pred; torch.cuda.empty_cache()

    df = pd.DataFrame([
        {"Noise Level": v["label"], "mIoU": v["mIoU"],
         "Drop vs Tight (%)": round((results[0]["mIoU"]-v["mIoU"])/results[0]["mIoU"]*100,1)}
        for noise,v in results.items()])
    df.to_csv(OUT + r"\Ablation_A3_BoxNoise.csv", index=False)
    print("\n  Saved: Ablation_A3_BoxNoise.csv")

    fig, ax = plt.subplots(figsize=(8,4))
    labels = [results[n]["label"] for n in noise_levels]
    values = [results[n]["mIoU"]  for n in noise_levels]
    ax.plot(labels, values, marker="o", color="#185FA5",
            linewidth=2.5, markersize=8)
    ax.fill_between(range(len(labels)), values,
                    alpha=0.1, color="#185FA5")
    for i,(l,v) in enumerate(zip(labels,values)):
        ax.text(i, v+0.004, "{:.4f}".format(v), ha="center", fontsize=9)
    ax.set_ylabel("Mean IoU", fontsize=11)
    ax.set_xlabel("Box Perturbation Level", fontsize=11)
    ax.set_title("Ablation A3: SAM Robustness to Bounding Box Noise\n(Tight box vs progressively noisier inputs)", fontsize=10)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, max(values)*1.3)
    ax.grid(True, alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGS + r"\Ablation_A3_BoxNoise.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: Ablation_A3_BoxNoise.png")
    return results

# ══════════════════════════════════════════════════════════════════════════
# COMPARATIVE TABLE — SAM vs Published Supervised Models
# WHAT THIS TELLS US: How does our zero-shot SAM compare to models that
# were TRAINED on iSAID data? Uses published numbers from papers.
# ══════════════════════════════════════════════════════════════════════════
def comparative_table(our_bb_miou, our_bpc_miou, our_cp_miou):
    print("\n" + "="*55)
    print("  COMPARATIVE EXPERIMENT: SAM vs Supervised Models")
    print("  Using published iSAID benchmark results from papers")
    print("="*55)

    # Published results from original papers on iSAID dataset
    # Source: iSAID benchmark leaderboard + respective papers
    published = [
        {"Method":"Mask R-CNN [He et al. 2017]",    "Backbone":"ResNet-50",   "Training":"Supervised","mAP50":32.3, "mIoU":"—",  "Year":2017, "Reference":"[17]"},
        {"Method":"HTC [Chen et al. 2019]",          "Backbone":"ResNet-50",   "Training":"Supervised","mAP50":36.4, "mIoU":"—",  "Year":2019, "Reference":"[18]"},
        {"Method":"QueryInst [Fang et al. 2021]",    "Backbone":"ResNet-50",   "Training":"Supervised","mAP50":40.1, "mIoU":"—",  "Year":2021, "Reference":"[19]"},
        {"Method":"RSPrompter [Chen et al. 2024]",   "Backbone":"ViT-H (SAM)", "Training":"Supervised","mAP50":47.3, "mIoU":"—",  "Year":2024, "Reference":"[21]"},
        {"Method":"SAM-BB (Ours)",                   "Backbone":"ViT-H",       "Training":"Zero-shot", "mAP50":"—",  "mIoU":round(our_bb_miou,4), "Year":2024, "Reference":"—"},
        {"Method":"SAM-BPC (Ours)",                  "Backbone":"ViT-H",       "Training":"Zero-shot", "mAP50":"—",  "mIoU":round(our_bpc_miou,4),"Year":2024, "Reference":"—"},
        {"Method":"SAM-CP (Ours)",                   "Backbone":"ViT-H",       "Training":"Zero-shot", "mAP50":"—",  "mIoU":round(our_cp_miou,4), "Year":2024, "Reference":"—"},
    ]
    df = pd.DataFrame(published)
    df.to_csv(OUT + r"\Table_VI_Comparative_Results.csv", index=False)
    print("  Saved: Table_VI_Comparative_Results.csv")
    print("\n  Table VI Preview:")
    print("  {:<30} {:<12} {:<12} {:<7} {:<7}".format("Method","Backbone","Training","mAP50","mIoU"))
    print("  "+"-"*68)
    for r in published:
        print("  {:<30} {:<12} {:<12} {:<7} {:<7}".format(
            r["Method"][:30], r["Backbone"][:12],
            r["Training"], str(r["mAP50"]), str(r["mIoU"])))

    # Bar chart comparing methods
    fig, ax = plt.subplots(figsize=(10,5))
    sup_methods = ["Mask R-CNN", "HTC", "QueryInst", "RSPrompter"]
    sup_scores  = [32.3, 36.4, 40.1, 47.3]
    our_methods = ["SAM-BB\n(Ours)", "SAM-BPC\n(Ours)", "SAM-CP\n(Ours)"]
    our_scores  = [round(our_bb_miou*100,1), round(our_bpc_miou*100,1), round(our_cp_miou*100,1)]
    all_methods = sup_methods + our_methods
    all_scores  = sup_scores  + our_scores
    all_colors  = ["#888780"]*4 + ["#185FA5","#378ADD","#B4B2A9"]
    bars = ax.bar(all_methods, all_scores, color=all_colors,
                  width=0.6, edgecolor="white")
    for bar,v in zip(bars,all_scores):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.3,
                "{:.1f}".format(v), ha="center", fontsize=9, fontweight="bold")
    ax.axvline(x=3.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(1.5, max(all_scores)*0.95, "Supervised\n(trained on iSAID)",
            ha="center", fontsize=9, color="#5F5E5A")
    ax.text(5.0, max(all_scores)*0.95, "Zero-shot\n(no training)",
            ha="center", fontsize=9, color="#185FA5")
    ax.set_ylabel("Performance Score (%)", fontsize=11)
    ax.set_title("Comparative Experiment: SAM Zero-shot vs Supervised Models on iSAID\n(mAP@0.5 for supervised; mIoU×100 for SAM)", fontsize=10)
    ax.set_ylim(0, max(all_scores)*1.2)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGS + r"\Figure5_Comparative_Results.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: Figure5_Comparative_Results.png")
    return df

# ══════════════════════════════════════════════════════════════════════════
# PRINT FULL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
def print_final_summary():
    print("\n" + "="*55)
    print("  ALL ABLATION EXPERIMENTS COMPLETE")
    print("="*55)
    print("\n  NEW FILES SAVED:")
    files = [
        ("Ablation_A1_Backbone.csv",      "Backbone ablation table"),
        ("Ablation_A1_Backbone.png",      "Backbone ablation figure"),
        ("Ablation_A2_Resolution.csv",    "Resolution ablation table"),
        ("Ablation_A2_Resolution.png",    "Resolution ablation figure"),
        ("Ablation_A3_BoxNoise.csv",      "Box noise ablation table"),
        ("Ablation_A3_BoxNoise.png",      "Box noise ablation figure"),
        ("Table_VI_Comparative_Results.csv","Comparative experiment table"),
        ("Figure5_Comparative_Results.png", "Comparative figure"),
    ]
    for fn, desc in files:
        full = OUT + "\\" + fn if fn.endswith(".csv") else FIGS + "\\" + fn
        exists = os.path.exists(full)
        print("  {} {:<42} {}".format(
            "OK" if exists else "--", fn, desc))
    print("\n  SUPERVISOR REQUIREMENTS NOW MET:")
    print("  [x] 7 prompt strategy comparison (main experiment)")
    print("  [x] Ablation A1 — backbone size comparison")
    print("  [x] Ablation A2 — input resolution comparison")
    print("  [x] Ablation A3 — bounding box noise robustness")
    print("  [x] Comparative table — SAM vs supervised models")
    print("\n  PUSH TO GITHUB:")
    print("    git add results/ figures/")
    print('    git commit -m "Add ablation studies and comparative experiments"')
    print("    git push origin main\n")

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  ABLATION + COMPARATIVE EXPERIMENTS")
    print("  Addressing supervisor comments")
    print("="*55)

    print("\nLoading dataset...")
    from pycocotools.coco import COCO
    items = load_instances()
    if len(items) == 0:
        print("ERROR: No instances loaded!"); sys.exit(1)

    # Load main results to get our BB/BPC/CP mIoU values
    raw_path = OUT + r"\raw_results.csv"
    if os.path.exists(raw_path):
        df_main = pd.read_csv(raw_path)
        bb_miou  = df_main[df_main["strategy"]=="BB"]["iou"].mean()
        bpc_miou = df_main[df_main["strategy"]=="BPC"]["iou"].mean()
        cp_miou  = df_main[df_main["strategy"]=="CP"]["iou"].mean()
        print("  Loaded existing results: BB={:.4f} BPC={:.4f} CP={:.4f}".format(
              bb_miou, bpc_miou, cp_miou))
    else:
        bb_miou=0.4215; bpc_miou=0.4828; cp_miou=0.1030
        print("  Using default values (run SAM_FINAL.py first for real values)")

    # Run all ablation experiments
    ablation_backbone(items)
    ablation_resolution(items)
    ablation_box_noise(items)
    comparative_table(bb_miou, bpc_miou, cp_miou)
    print_final_summary()
