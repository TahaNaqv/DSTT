Assignment #01 - Data Science Tools & Techniques (MS DS)
Q#1: Data Pre-processing and Exploratory Data Analysis

Name    : Taha Abbas Naqvi
Roll No : 261-8004
Class   : MS (DS)
Section : A

Files
  Assignment1_Q1_Preprocessing_EDA.ipynb   main deliverable, already run, all outputs saved
  Report.md / Report.pdf                   the short report
  Taha Naqvi - The Attached Dataset 2.csv  dataset 2, as provided with the assignment
  output/                                  cleaned datasets written by the last cells

Datasets
  1. Census-Income (KDD): https://archive.ics.uci.edu/dataset/117/census+income+kdd
     The notebook downloads it into data/ the first time it runs and reuses that copy
     afterwards, so data/ is not part of the submission.
  2. The attached csv, included here.

How to run
  pip install pandas numpy matplotlib seaborn scikit-learn jupyter
  jupyter lab Assignment1_Q1_Preprocessing_EDA.ipynb    then Run All

  Takes about a minute, plus one 10 MB download the first time.
  Tested on Python 3.12 with pandas 3.0.5, scikit-learn 1.9.0, seaborn 0.13.2.

Notebook layout
  Part A   dataset 2: EDA before cleaning, issue list, cleaning, EDA after
  Part B   dataset 1: same sequence
  Part C   both datasets compared before and after, cleaned files written out
