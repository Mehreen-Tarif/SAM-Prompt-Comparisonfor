\# 3. Experimental Methodology



\## 3.1 Experimental Design



Our comprehensive experimental framework evaluates seven prompt strategies across 200 remote sensing images spanning 10 object classes (Table 1). The systematic workflow (Figure 1) ensures reproducible and statistically robust comparisons.



\*\*Research Questions:\*\*

1\. RQ1: Which prompt strategy maximizes segmentation accuracy (IoU) across diverse remote sensing objects?

2\. RQ2: How does prompt complexity affect the trade-off between accuracy and computational efficiency?

3\. RQ3: Are certain strategies more robust to object morphological variations?

4\. RQ4: What statistical confidence can we attribute to observed performance differences?



\## 3.2 Experimental Setup



\### 3.2.1 Hardware Configuration

\- \*\*GPU\*\*: NVIDIA RTX 3090 (24GB VRAM)

\- \*\*CPU\*\*: Intel Xeon Gold 6226R

\- \*\*RAM\*\*: 128GB DDR4

\- \*\*Storage\*\*: 2TB NVMe SSD



\### 3.2.2 Software Environment

\- \*\*OS\*\*: Ubuntu 20.04 LTS

\- \*\*Deep Learning Framework\*\*: PyTorch 2.5.1

\- \*\*CUDA\*\*: 11.8

\- \*\*Python\*\*: 3.10.0

\- \*\*SAM Version\*\*: ViT-H with default weights



\### 3.2.3 Prompt Strategies Evaluated

1\. \*\*Center Point (CP)\*\*: Single point at object centroid

2\. \*\*Bounding Box (BB)\*\*: Tight bounding box annotation

3\. \*\*Multiple Points (MP3)\*\*: Three randomly sampled interior points

4\. \*\*Multiple Points (MP5)\*\*: Five randomly sampled interior points

5\. \*\*Multiple Points (MP10)\*\*: Ten randomly sampled interior points

6\. \*\*Box with Center (BBC)\*\*: Bounding box + center point

7\. \*\*Points with Box (PWB)\*\*: Bounding box + five corner/center points



\## 3.3 Evaluation Metrics



We employ comprehensive evaluation metrics:



1\. \*\*Intersection over Union (IoU)\*\*:

&nbsp;  $$IoU = \\frac{|P \\cap G|}{|P \\cup G|}$$



2\. \*\*Dice Coefficient\*\*:

&nbsp;  $$Dice = \\frac{2|P \\cap G|}{|P| + |G|}$$



3\. \*\*Precision, Recall, F1-Score\*\*:

&nbsp;  $$Precision = \\frac{TP}{TP + FP}, \\quad Recall = \\frac{TP}{TP + FN}$$

&nbsp;  $$F1 = 2 \\cdot \\frac{Precision \\cdot Recall}{Precision + Recall}$$



4\. \*\*Hausdorff Distance\*\*:

&nbsp;  $$H(P,G) = \\max\\left\\{\\sup\_{p\\in P}\\inf\_{g\\in G}d(p,g), \\sup\_{g\\in G}\\inf\_{p\\in P}d(p,g)\\right\\}$$



5\. \*\*Inference Time\*\*: Processing time per image (seconds)



6\. \*\*Confidence Score\*\*: SAM's internal confidence metric \[0,1]



\## 3.4 Statistical Analysis Protocol



\### 3.4.1 Descriptive Statistics

\- Mean ± standard deviation for each metric

\- 95% confidence intervals via bootstrapping (n=1000)



\### 3.4.2 Inferential Statistics

1\. \*\*One-way ANOVA\*\*: Test overall strategy differences

2\. \*\*Paired t-tests\*\*: Pairwise comparisons with Bonferroni correction

3\. \*\*Cohen's d\*\*: Effect size calculation for practical significance

4\. \*\*Win Rate Analysis\*\*: Percentage each strategy outperforms others



\### 3.4.3 Robustness Analysis

\- Performance variation across object classes

\- Sensitivity to object size and aspect ratio

\- Computational efficiency vs. accuracy trade-off



\## 3.5 Implementation Details



All experiments used fixed random seed (42) for reproducibility. Each strategy was evaluated independently with:

\- Batch size: 1 (sequential processing)

\- Image resolution: Preserved original (1024×1024 typical)

\- SAM parameters: Default settings, multimask\_output=False

\- Warm-up iterations: 10 (excluded from timing)



\## 3.6 Dataset Characteristics



| Class | Samples | Size Range (px²) | Aspect Ratio | Complexity |

|-------|---------|------------------|--------------|------------|

| Building | 40 | 500-5000 | 0.8-1.2 | Medium |

| Road | 35 | 1000-10000 | 3.0-10.0 | High |

| Vehicle | 30 | 50-500 | 1.5-2.5 | Low |

| Vegetation | 25 | 1000-20000 | 1.0-3.0 | High |

| Water | 20 | 2000-15000 | 1.0-5.0 | Medium |

| Bridge | 15 | 300-3000 | 2.0-6.0 | High |

| Airport | 10 | 5000-50000 | 1.0-2.0 | Medium |

| Ship | 10 | 200-2000 | 2.0-4.0 | Low |

| Stadium | 10 | 1000-10000 | 1.0-1.5 | Medium |

| Industrial | 5 | 2000-10000 | 1.0-3.0 | High |



\*\*Total\*\*: 200 samples across 10 classes

