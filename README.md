SAM Prompt Strategy Comparison on iSAID
Code and data release for the paper:
"Segment Anything Model for Remote Sensing Instance Segmentation: Systematic Prompt Strategy Evaluation and Bounding Box Superiority on iSAID"
Mehreen Tarif, Wu Xu, Sana Abbas  
College of Computer Science and Cyber Security  
Chengdu University of Technology, China
Submitted to IEEE Geoscience and Remote Sensing Letters (GRSL), 2026
---
Overview
This repository contains the experimental framework for evaluating seven prompt configurations of the Segment Anything Model (SAM) on the iSAID aerial imagery benchmark. The main finding is that box-based prompts substantially outperform point-based prompts, with Box-plus-Center (BPC) achieving 0.432 mIoU compared to 0.111 for Center Point (Cohen's d = 1.49, p < 0.001 after Bonferroni correction).
Key Results
Strategy	mIoU	95% CI	F1
BPC (Box-plus-Center)	0.433	[0.405, 0.461]	0.598
BB (Bounding Box)	0.374	[0.347, 0.402]	0.510
MP10 (Multi-Point 10)	0.154	[0.136, 0.173]	0.272
MP5 (Multi-Point 5)	0.135	[0.119, 0.152]	0.244
MP3 (Multi-Point 3)	0.122	[0.108, 0.137]	0.224
CP (Center Point)	0.112	[0.098, 0.127]	0.211
CPC (Corner-Points + Center)	0.099	[0.087, 0.112]	0.182
Evaluation: 333 iSAID instances across 15 categories, 2,331 SAM inferences, SAM ViT-H at 1024x1024 resolution, fixed random seed = 42.
Repository Structure
```
SAM-Prompt-Comparisonfor/
├── README.md                       # This file
├── master_experiment.py            # Main experiment script
├── build_fig12_clean.py            # Qualitative figure generation
├── requirements.txt                # Python dependencies
├── results/                        # Pre-computed results
│   ├── raw_results_full15.csv
│   ├── Table_II_Overall.csv
│   ├── Table_III_Stats.csv
│   ├── Ablation_A1_Backbone.csv
│   ├── Ablation_A2_Resolution.csv
│   └── Ablation_A3_BoxNoise.csv
└── figures/                        # Paper figures
```
Setup
Requirements
Python 3.8+
CUDA-capable GPU (RTX 3090 used in paper)
About 20 GB disk space for iSAID dataset
Installation
```bash
git clone https://github.com/Mehreen-Tarif/SAM-Prompt-Comparisonfor.git
cd SAM-Prompt-Comparisonfor

pip install torch torchvision
pip install opencv-python pandas numpy tqdm
pip install pycocotools matplotlib scipy
pip install git+https://github.com/facebookresearch/segment-anything.git
```
Download SAM checkpoints
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
Download iSAID dataset
iSAID is available at https://captain-whu.github.io/iSAID/
Download train and val splits with COCO-format annotations.
Running the Experiments
Update the BASE path at the top of master_experiment.py to point to your local iSAID directory, then run:
```bash
python master_experiment.py
```
This will run:
Phase 1: Build experiment plan (333 instances, 15 categories)
Phase 2: Main experiment (7 strategies x 333 instances)
Phase 3: Ablation A1 (ViT-H vs ViT-B backbone)
Phase 4: Ablation A2 (Resolution 512/768/1024)
Phase 5: Ablation A3 (Bounding box noise)
Phase 6: Statistical analysis
Expected runtime: 75-90 minutes on RTX 3090.
The script is resumable. If interrupted, re-run and it continues from the last checkpoint.
Reproducing the Paper's Results
After running master_experiment.py, the output CSVs match the numbers reported in the paper:
Table II in the paper matches Table_II_Overall.csv
Table III in the paper matches Table_III_Stats.csv
Ablation A1/A2/A3 match Section IV-C of the paper
Fixed random seed (42) ensures reproducibility across runs.
Sampling Note
The 333 evaluation instances are the 25 largest instances per category. This choice ensures prompt-mask correspondence is meaningful and visible. Performance on smaller or occluded instances may differ. See the Limitations section of the paper for discussion.
Citation
If you use this code or build on this work, please cite:
```
@article{tarif2026sam,
  title={Segment Anything Model for Remote Sensing Instance Segmentation: 
         Systematic Prompt Strategy Evaluation and Bounding Box Superiority on iSAID},
  author={Tarif, Mehreen and Xu, Wu and Abbas, Sana},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2026},
  note={Submitted}
}
```
Acknowledgments
iSAID benchmark by the CAPTAIN laboratory, Wuhan University
Segment Anything Model by Meta AI Research
DOTA dataset (which iSAID builds upon)
Contact
First author (Mehreen Tarif): mehreentarif17@gmail.com
Corresponding author (Wu Xu): wuxu2022@cdut.edu.cn
License
This code is released for research and reproducibility purposes. Please contact the authors before commercial use.
