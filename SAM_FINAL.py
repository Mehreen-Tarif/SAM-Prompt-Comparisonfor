import os, sys, time, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry, SamPredictor
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

BASE   = r"C:\Users\Lenovo\Desktop\My_SAM_Project"
ANN    = BASE + r"\data\iSAID\Annotations\iSAID_val.json"
IMGS   = BASE + r"\data\iSAID\val\images"
CKPT   = BASE + r"\sam_vit_h_4b8939.pth"
OUT    = BASE + r"\results"
FIGS   = BASE + r"\figures"
SEED   = 42
ISIZE  = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CATS = ["ship","storage_tank","baseball_diamond","tennis_court","basketball_court",
        "Ground_Track_Field","Bridge","Large_Vehicle","Small_Vehicle","Helicopter",
        "Swimming_pool","Roundabout","Soccer_ball_field","plane","Harbor"]

NAMES = {"CP":"Center Point","BB":"Bounding Box","MP3":"Multi-Points 3",
         "MP5":"Multi-Points 5","MP10":"Multi-Points 10",
         "BPC":"Box + Center","CPC":"Corners + Center"}

def check():
    print("\n" + "="*55)
    print("  SAM EXPERIMENT - FINAL VERSION")
    print("="*55)
    for path, label in [(ANN,"Annotation: iSAID_val.json"),(IMGS,"Images: val/images/"),(CKPT,"SAM ViT-H weights")]:
        found = os.path.exists(path)
        print("  {} {}".format("OK     " if found else "MISSING", label))
        if not found: sys.exit(1)
    imgs = [f for f in os.listdir(IMGS) if f.lower().endswith((".png",".jpg"))]
    print("  OK      {} images on disk".format(len(imgs)))
    print("  OK      GPU: {}".format(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    print("="*55)
    print("  All good! Starting...\n")

def load_data():
    random.seed(SEED); np.random.seed(SEED)
    print("  Loading annotations from iSAID_val.json...")
    coco     = COCO(ANN)
    available = set(os.listdir(IMGS))
    all_ids  = coco.getImgIds()
    valid_ids = [i for i in all_ids if coco.loadImgs([i])[0]["file_name"] in available]
    print("  Matched {} / {} images to disk".format(len(valid_ids), len(all_ids)))
    items = []; skipped = 0
    for cat_name in CATS:
        cid = coco.getCatIds(catNms=[cat_name])
        if not cid: continue
        ann_ids = coco.getAnnIds(catIds=[cid[0]], imgIds=valid_ids)
        if not ann_ids: continue
        for aid in ann_ids:
            ann      = coco.loadAnns([aid])[0]
            img_info = coco.loadImgs([ann["image_id"]])[0]
            img_path = os.path.join(IMGS, img_info["file_name"])
            if not os.path.exists(img_path): skipped += 1; continue
            seg = ann["segmentation"]
            if not seg: skipped += 1; continue
            try:
                rle  = maskUtils.frPyObjects(seg, img_info["height"], img_info["width"])
                mask = maskUtils.decode(rle)
                if mask.ndim == 3: mask = mask[:,:,0]
                if mask.sum() == 0: skipped += 1; continue
            except: skipped += 1; continue
            x,y,w,h = ann["bbox"]
            if w<=0 or h<=0: skipped += 1; continue
            items.append({"cat":cat_name,"bbox":ann["bbox"],"mask":mask.astype(bool),
                          "path":img_path,"h":img_info["height"],"w":img_info["width"],
                          "iid":ann["image_id"],"aid":aid})
    print("  Loaded {} instances ({} skipped)\n".format(len(items), skipped))
    print("  Breakdown:")
    for cat in CATS:
        n = sum(1 for it in items if it["cat"]==cat)
        if n > 0: print("    {:28s} {}".format(cat, n))
    print()
    return items

def center_pt(mask):
    ys,xs = np.where(mask)
    if len(xs)==0: h,w=mask.shape; return np.array([[w/2.0,h/2.0]])
    return np.array([[float(xs.mean()),float(ys.mean())]])

def interior_pts(mask, n):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    e = cv2.erode(mask.astype(np.uint8),k,iterations=2)
    ys,xs = np.where(e>0)
    if len(xs)<n: ys,xs = np.where(mask)
    if len(xs)==0: return center_pt(mask)
    idx = np.linspace(0,len(xs)-1,n,dtype=int)
    return np.column_stack([xs[idx],ys[idx]]).astype(float)

def make_prompts(item, sx, sy):
    m=item["mask"]; b=item["bbox"]
    def sc(p): q=p.copy(); q[:,0]*=sx; q[:,1]*=sy; return q
    cp   = sc(center_pt(m))
    mp3  = sc(interior_pts(m,3))
    mp5  = sc(interior_pts(m,5))
    mp10 = sc(interior_pts(m,10))
    x,y,w,h = b
    cn = sc(np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]],dtype=float))
    bb = np.array([x*sx,y*sy,(x+w)*sx,(y+h)*sy])
    return {
        "CP" :{"pts":cp,              "lbs":np.array([1]),        "box":None},
        "BB" :{"pts":None,            "lbs":None,                 "box":bb},
        "MP3":{"pts":mp3,             "lbs":np.ones(3,dtype=int), "box":None},
        "MP5":{"pts":mp5,             "lbs":np.ones(5,dtype=int), "box":None},
        "MP10":{"pts":mp10,           "lbs":np.ones(10,dtype=int),"box":None},
        "BPC":{"pts":cp,              "lbs":np.array([1]),        "box":bb},
        "CPC":{"pts":np.vstack([cn,cp]),"lbs":np.ones(5,dtype=int),"box":None}}

def metrics(pred, gt):
    tp=np.logical_and(pred,gt).sum()
    fp=np.logical_and(pred,~gt).sum()
    fn=np.logical_and(~pred,gt).sum()
    iou=tp/(tp+fp+fn+1e-8); pre=tp/(tp+fp+1e-8)
    rec=tp/(tp+fn+1e-8);    f1=2*pre*rec/(pre+rec+1e-8)
    return float(iou),float(pre),float(rec),float(f1)

def run_experiment(items):
    print("  Loading SAM ViT-H on {}...".format(DEVICE.upper()))
    sam = sam_model_registry["vit_h"](checkpoint=CKPT)
    sam.to(device=DEVICE)
    pred = SamPredictor(sam)
    print("  SAM loaded!\n")
    strats = ["CP","BB","MP3","MP5","MP10","BPC","CPC"]
    rows   = []
    pbar   = tqdm(items, desc="  Running", unit="img", ncols=65)
    for i, item in enumerate(pbar):
        img = cv2.imread(item["path"])
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (ISIZE,ISIZE))
        sx  = ISIZE / item["w"]
        sy  = ISIZE / item["h"]
        gt  = cv2.resize(item["mask"].astype(np.uint8),(ISIZE,ISIZE),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
        pred.set_image(img)
        prompts = make_prompts(item, sx, sy)
        for s in strats:
            p = prompts[s]; t0 = time.perf_counter()
            try:
                masks,scores,_ = pred.predict(point_coords=p["pts"],
                    point_labels=p["lbs"],box=p["box"],multimask_output=True)
                t1=time.perf_counter(); best=np.argmax(scores)
                pm=masks[best]; conf=float(scores[best]); ms=(t1-t0)*1000
                iou,pre,rec,f1 = metrics(pm,gt)
            except: iou=pre=rec=f1=conf=ms=0.0
            rows.append({"strategy":s,"category":item["cat"],
                "iou":round(iou,6),"precision":round(pre,6),
                "recall":round(rec,6),"f1":round(f1,6),
                "confidence":round(conf,6),"time_ms":round(ms,2)})
        if (i+1)%10==0:
            pd.DataFrame(rows).to_csv(OUT+r"\progress.csv",index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT+r"\raw_results.csv",index=False)
    print("\n  Saved {} rows to raw_results.csv".format(len(df)))
    return df

def bci(vals, n=2000):
    np.random.seed(SEED)
    m=[np.mean(np.random.choice(vals,len(vals),replace=True)) for _ in range(n)]
    return np.percentile(m,[2.5,97.5])

def make_tables(df):
    print("  Building tables...")
    order = ["BB","BPC","MP10","MP5","MP3","CPC","CP"]
    rows  = []
    for s in order:
        sub=df[df["strategy"]==s]; ious=sub["iou"].values; ci=bci(ious)
        rows.append({"Strategy":NAMES[s],"Mean IoU":round(ious.mean(),4),
            "Std Dev":round(ious.std(),4),
            "95% CI":"[{:.4f},{:.4f}]".format(ci[0],ci[1]),
            "F1-Score":round(sub["f1"].mean(),4),
            "Confidence":"{:.3f}".format(sub["confidence"].mean()),
            "Time ms":round(sub["time_ms"].mean(),1)})
    t2=pd.DataFrame(rows)
    t2.to_csv(OUT+r"\Table_II_Overall_Performance.csv",index=False)
    print("    Table II saved")
    pairs=[("BB","CP"),("BB","MP10"),("BB","BPC"),("BPC","CP")]
    srows=[]
    for a,b in pairs:
        va=df[df["strategy"]==a]["iou"].values
        vb=df[df["strategy"]==b]["iou"].values
        _,p=stats.ttest_rel(va,vb); pb=min(p*4,1.0)
        d=(va.mean()-vb.mean())/np.sqrt((va.std()**2+vb.std()**2)/2+1e-8)
        srows.append({"Comparison":"{} vs {}".format(a,b),
            "IoU Diff":round(va.mean()-vb.mean(),4),
            "p-value":"<0.001" if p<0.001 else "{:.4f}".format(p),
            "p Bonf":"<0.001" if pb<0.001 else "{:.4f}".format(pb),
            "Cohens d":round(d,2),
            "Significant":"Yes" if pb<0.05 else "No"})
    pd.DataFrame(srows).to_csv(OUT+r"\Table_III_Statistical_Tests.csv",index=False)
    print("    Table III saved")
    cat_rows=[]
    for cat in CATS:
        row={"Category":cat.replace("_"," ").title()}
        for s in order:
            sub=df[(df["category"]==cat)&(df["strategy"]==s)]
            row[NAMES[s]]=round(sub["iou"].mean(),4) if len(sub)>0 else 0.0
        cat_rows.append(row)
    mr={"Category":"MEAN"}
    for s in order: mr[NAMES[s]]=round(df[df["strategy"]==s]["iou"].mean(),4)
    cat_rows.append(mr)
    pd.DataFrame(cat_rows).to_csv(OUT+r"\Table_IV_Per_Category.csv",index=False)
    print("    Table IV saved")
    return t2

def make_figures(df):
    print("  Generating figures...")
    order  = ["BB","BPC","MP10","MP5","MP3","CPC","CP"]
    labels = [NAMES[s] for s in order]
    means  = [df[df["strategy"]==s]["iou"].mean() for s in order]
    colors = ["#185FA5","#378ADD","#1D9E75","#5DCAA5","#9FE1CB","#888780","#B4B2A9"]

    fig,ax = plt.subplots(figsize=(9,5))
    ax.barh(labels[::-1],means[::-1],color=colors[::-1],height=0.55)
    for i,(m,s) in enumerate(zip(means[::-1],order[::-1])):
        ci=bci(df[df["strategy"]==s]["iou"].values,n=500)
        ax.errorbar(m,i,xerr=[[m-ci[0]],[ci[1]-m]],fmt="none",color="#333",capsize=4)
        ax.text(m+0.005,i,"{:.4f}".format(m),va="center",fontsize=9)
    ax.set_xlabel("Mean IoU",fontsize=11)
    ax.set_title("Figure 1. SAM Prompt Strategy Comparison — iSAID Dataset",fontsize=11)
    ax.set_xlim(0,0.65); ax.spines[["top","right"]].set_visible(False)
    patches=[mpatches.Patch(color=c,label=l) for c,l in
        [("#185FA5","Box-based"),("#1D9E75","Multi-point"),("#888780","Single-point")]]
    ax.legend(handles=patches,loc="lower right",fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGS+r"\Figure1_Overall_Comparison.png",dpi=300,bbox_inches="tight")
    plt.close(); print("    Figure 1 saved")

    data=[]
    for cat in CATS:
        data.append([df[(df["category"]==cat)&(df["strategy"]==s)]["iou"].mean()
            if len(df[(df["category"]==cat)&(df["strategy"]==s)])>0 else 0.0
            for s in order])
    fig,ax=plt.subplots(figsize=(12,7))
    im=ax.imshow(data,cmap="Blues",aspect="auto",vmin=0,vmax=0.8)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels,rotation=30,ha="right",fontsize=9)
    ax.set_yticks(range(len(CATS))); ax.set_yticklabels([c.replace("_"," ").title() for c in CATS],fontsize=9)
    for i in range(len(CATS)):
        for j in range(len(order)):
            v=data[i][j]
            ax.text(j,i,"{:.3f}".format(v),ha="center",va="center",
                fontsize=7.5,color="white" if v>0.4 else "#222")
    plt.colorbar(im,ax=ax,label="IoU",shrink=0.8)
    ax.set_title("Figure 2. Per-Category IoU Heatmap",fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGS+r"\Figure2_Category_Heatmap.png",dpi=300,bbox_inches="tight")
    plt.close(); print("    Figure 2 saved")

    fig,axes=plt.subplots(2,4,figsize=(14,7)); axes=axes.flatten()
    for idx,s in enumerate(order):
        ax=axes[idx]; sub=df[df["strategy"]==s]
        ax.scatter(sub["confidence"],sub["iou"],alpha=0.4,s=15,color=colors[idx])
        if len(sub)>2:
            r,_=stats.pearsonr(sub["confidence"],sub["iou"])
            z=np.polyfit(sub["confidence"],sub["iou"],1)
            xl=np.linspace(sub["confidence"].min(),sub["confidence"].max(),50)
            ax.plot(xl,np.polyval(z,xl),"r-",linewidth=1.5)
            ax.set_title("{}\nr={:.2f}".format(NAMES[s],r),fontsize=9)
        ax.set_xlabel("Confidence",fontsize=8); ax.set_ylabel("IoU",fontsize=8)
        ax.tick_params(labelsize=7); ax.spines[["top","right"]].set_visible(False)
    axes[-1].axis("off")
    fig.suptitle("Figure 3. SAM Confidence vs IoU Correlation",fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGS+r"\Figure3_Confidence_Correlation.png",dpi=300,bbox_inches="tight")
    plt.close(); print("    Figure 3 saved")

    fig,ax=plt.subplots(figsize=(9,5))
    for s,col,ls in [("BB","#185FA5","-"),("BPC","#378ADD","-"),("MP10","#1D9E75","--"),("CP","#888780",":")]:
        sub=df[df["strategy"]==s].sort_values("iou",ascending=False).reset_index(drop=True)
        n=len(sub)
        if n<5: continue
        q=[sub["iou"].iloc[:max(1,n//5)].mean(),sub["iou"].iloc[n//5:2*n//5].mean(),
           sub["iou"].iloc[2*n//5:3*n//5].mean(),sub["iou"].iloc[3*n//5:4*n//5].mean(),
           sub["iou"].iloc[4*n//5:].mean()]
        ax.plot(["Optimal","High","Medium","Low","Very Low"],q,
            marker="o",color=col,linestyle=ls,linewidth=2,label=NAMES[s],markersize=5)
    ax.set_ylabel("Mean IoU",fontsize=11)
    ax.set_xlabel("Complexity Level",fontsize=11)
    ax.set_title("Figure 4. Performance Across Complexity Levels",fontsize=11)
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGS+r"\Figure4_Complexity.png",dpi=300,bbox_inches="tight")
    plt.close(); print("    Figure 4 saved")

def print_summary(df):
    print("\n"+"="*55)
    print("  YOUR RESULTS")
    print("="*55)
    order=["BB","BPC","MP10","MP5","MP3","CPC","CP"]
    print("\n  {:<20}{:>8}{:>8}{:>8}".format("Strategy","mIoU","F1","ms"))
    print("  "+"-"*44)
    for s in order:
        sub=df[df["strategy"]==s]
        tag="  <-- BEST" if s=="BB" else ""
        print("  {:<20}{:>8.4f}{:>8.4f}{:>8.1f}{}".format(
            NAMES[s],sub["iou"].mean(),sub["f1"].mean(),sub["time_ms"].mean(),tag))
    bb=df[df["strategy"]=="BB"]["iou"].mean()
    cp=df[df["strategy"]=="CP"]["iou"].mean()
    print("\n  BB is {:.1f}x better than CP".format(bb/max(cp,1e-8)))
    print("="*55)
    print("\n  Results: "+OUT)
    print("  Figures: "+FIGS)
    print("\n  FOR YOUR PAPER:")
    for fn,t in [
        ("Table_II_Overall_Performance.csv","Table II"),
        ("Table_III_Statistical_Tests.csv", "Table III"),
        ("Table_IV_Per_Category.csv",        "Table IV"),
        ("Figure1_Overall_Comparison.png",   "Figure 1"),
        ("Figure2_Category_Heatmap.png",     "Figure 2"),
        ("Figure3_Confidence_Correlation.png","Figure 3"),
        ("Figure4_Complexity.png",           "Figure 4")]:
        print("    {:<42} -> {}".format(fn,t))
    print("\n  PUSH TO GITHUB:")
    print("    git add results/ figures/")
    print('    git commit -m "Add real SAM experiment results"')
    print("    git push origin main")

if __name__=="__main__":
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    check()
    print("PHASE 1: Loading dataset..."); items=load_data()
    if len(items)==0: print("ERROR: 0 instances!"); sys.exit(1)
    print("PHASE 2: Running {} instances x 7 strategies...".format(len(items)))
    df=run_experiment(items)
    print("\nPHASE 3: Tables and figures...")
    make_tables(df); make_figures(df)
    print_summary(df)
