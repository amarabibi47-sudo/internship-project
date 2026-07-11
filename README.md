# Internship Project

## Overview
This repository contains the work completed during my internship, including data analysis, notebooks, and source code developed as part of the internship program. The project focuses on building an image captioning pipeline using the Flickr8k dataset, covering data exploration, preprocessing, and model development.

## Goals
- Set up a proper development environment and workflow
- Learn and apply data analysis / programming skills
- Explore and preprocess the Flickr8k dataset (text and image)
- Deliver project milestones as per internship timeline
- Maintain clean, well-documented code and Git workflow practices

## Folder Structure
internship-project/
│
├── data/                  # Raw and processed datasets
│   ├── Images/             # Original Flickr8k images
│   ├── captions.txt        # Original captions file
│   └── processed/          # Cleaned captions, vocabulary, resized images
├── notebooks/              # Jupyter notebooks for exploration and analysis
├── src/                    # Source code and reusable scripts
├── tests/                  # Unit tests for source code
├── requirements.txt
└── README.md

## Setup Instructions
1. Clone the repository:
git clone https://github.com/amarabibi47-sudo/internship-project.git
cd internship-project
2. Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate
3. Install dependencies:
pip install -r requirements.txt

## Progress Log

### Day 1 — Environment Setup
- Set up Python virtual environment
- Installed Git and configured GitHub repository access
- Created repository structure (`data/`, `notebooks/`, `src/`, `tests/`)
- Verified environment with successful package installation

### Day 2 — Flickr8k Dataset Exploration
- Explored dataset: image count, captions per image, caption length distribution, vocabulary size
- Visual inspection of sample images with captions
- Identified data quality issues (duplicate captions, short/blank captions, corrupted images)
- Deliverable: `notebooks/01_flickr8k_exploration.ipynb` with statistics and plots

### Day 3 — Environment & Git Workflow Training
- Installed and tested core libraries: NLTK, spaCy, Hugging Face Transformers & Datasets, OpenCV, torchvision
- Practiced Git workflow: branching, commits, pull requests, and code review etiquette
- Loaded and previewed Flickr8k dataset in a notebook
- Deliverable: `requirements.txt` committed, first PR merged, dataset loading notebook

### Day 4 — Text & Image Preprocessing
- Text preprocessing on captions: tokenization, lowercasing, stop-word handling, vocabulary building
- Image preprocessing: resizing, normalization, channel consistency checks
- Deliverable: `notebooks/03_preprocessing.ipynb` producing cleaned/tokenized captions and resized/normalized images saved to `data/processed/`

## Status
🚧 In Progress — Day 4 completed: Text and image preprocessing pipeline built.

## Author
Amara Bibi

## Mentor
Mateen Yaqoob