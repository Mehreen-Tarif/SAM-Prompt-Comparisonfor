\# 🛰️ SAM Remote Sensing Prompt Study



<div align="center">



\[!\[Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



\*\*Official code for: "Optimal Prompt Strategy Selection for Segment Anything Model in Remote Sensing"\*\*



\*First comprehensive comparison of SAM prompt strategies on aerial imagery\*



</div>



---



\## 📖 \*\*What This Project Does\*\*



This repository contains code for our research paper that answers a simple but important question:



> \*\*"What's the BEST way to tell SAM what to segment in satellite/aerial images?"\*\*



We tested 7 different ways (called "prompts") and found that \*\*bounding boxes are 11× better than points\*\*! 🎯



| Prompt Type | Mean IoU | Speed | Best For |

|------------|----------|-------|----------|

| 🟦 \*\*Bounding Box\*\* | \*\*0.4594\*\* | ⚡ 8.8ms | Everything! |

| 🔵 Box + Center | 0.4439 | 9.8ms | Complex shapes |

| 🔴 10 Points | 0.1999 | 9.5ms | Simple objects |

| ⚫ 5 Points | 0.0712 | 9.7ms | Testing only |

| ⚪ 3 Points | 0.0486 | 12.6ms | Testing only |

| ✳️ Corners + Center | 0.0426 | 9.3ms | Testing only |

| ● Center Point | 0.0403 | 33.8ms | Testing only |



\*Table: Bounding boxes are clearly the winner!\*



---



\## 🚀 \*\*Quick Start (5 Minutes)\*\*



\### 1. \*\*Clone Repository\*\*

```bash

git clone https://github.com/your-username/SAM-RemoteSensing-Prompt-Study.git

cd SAM-RemoteSensing-Prompt-Study




## Clone with SSH (recommended if HTTPS fails):
```bash
git clone git@github.com:Mehreen-Tarif/SAM-Prompt-Comparison-.git

HTTPS: https://github.com/Mehreen-Tarif/SAM-Prompt-Comparison-.git

SSH: git@github.com:Mehreen-Tarif/SAM-Prompt-Comparison-.git


If you get "Connection was reset" error, use SSH instead of HTTPS.


