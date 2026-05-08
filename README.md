STAT 4052 Final Project: Highway Fuel Efficiency
This repository contains the R code for the STAT 4052 final project analyzing highway fuel efficiency using the fueleconomy::vehicles dataset.

File
ProjectCode — main analysis script containing all data cleaning, Box-Cox transformation, train/test split, and three regression models (MLR, PCR, Random Forest)

Requirements
The following R packages are required: MASS, fueleconomy, tidyverse, pls, randomForest, Metrics
Install with: install.packages(c("MASS", "fueleconomy", "tidyverse", "pls", "randomForest", "Metrics"))

Usage
Run ProjectCode top to bottom with set.seed(42) already set for reproducibility.
