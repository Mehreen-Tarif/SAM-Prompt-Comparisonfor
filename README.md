# SAM Experiment on iSAID Dataset

This repository contains code for experimenting with Segment Anything Model (SAM) on the iSAID aerial imagery dataset.

## ?? Project Overview

* Evaluate SAM's zero-shot performance on iSAID dataset
* Experiment with different prompting strategies
* Generate comprehensive analysis and paper figures

## ?? Quick Start

### 1\. Clone Repository

\\\\ash
git clone https://github.com/Mehreen-Tarif/SAM-Prompt-Comparison-.git
cd SAM-Prompt-Comparison-
\\\\

### 2\. Install Dependencies

\\\\ash
pip install -r requirements.txt
\\\\

### 3\. Quick Test

\\\\ash
python simple\_sam\_test.py
\\  
*(Uses test\_small.jpg included in repository)*

## ?? Full Experiment Setup

### 1\. Download iSAID Dataset

* Register at [iSAID Official Website](https://captain-whu.github.io/iSAID/)
* Download the dataset (~10GB)
* Extract to: \\data/iSAID/\\

### 2\. Download SAM Weights

\\\\ash

# SAM Base model (375MB)

wget https://dl.fbaipublicfiles.com/segment\_anything/sam\_vit\_b\_01ec64.pth -P models/
\\\\

### 3\. Expected Structure

\\  
My\_SAM\_Project/
 data/
    iSAID/
        train/
           images/
           masks/
        val/
           images/
           masks/
        test/
            images/
 models/
    sam\_vit\_b\_01ec64.pth
 \[all code files]
\\\\

### 4\. Run Complete Experiment

\\\\ash
python complete\_experiment.py
\\\\

## ?? Main Scripts

* \\complete\_experiment.py\\ - Run full experiment pipeline
* &nbsp; 
  eal\_isaid\_experiment.py\\ - Main experiment on iSAID
* &nbsp; 
  un\_\*.py\\ - Various experiment runners
* \\generate\_paper\_materials.py\\ - Generate figures/tables for paper

## ?? Repository Structure

* \\scripts/\\ - Utility scripts for data processing
* \\	est\_small.jpg\\ - Sample image for testing
* \\	able\_results.tex\\ - LaTeX table for paper results

## ?? Citation

If you use this code, please cite:
\\\\ibtex
@article{iSAID2019,
title={iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images},
author={Zamir, Syed Waqas and Arora, Aditya and Gupta, Akshita and Khan, Salman and Sun, Guolei and Khan, Fahad Shahbaz and Zhu, Fan and Shao, Ling},
journal={CVPRW},
year={2019}
}
\\\\

## ?? License

MIT License

