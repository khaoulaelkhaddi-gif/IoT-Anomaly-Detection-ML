# IoT-Anomaly-Detection-ML
Detecting Anomalous Behaviour in IoT Networks Using Simulated Environments and Machine Learning




## 1. Input  

Start with one dataset:  
- **full_public_dataset.csv** (public ToN-IoT)  
- **full_private_dataset.csv** (private Cooja)  



## 2. Pre-processing And Feature Selection  

Run the preprocessing script for each technique (**RF, Chi, MI**). Each script:  
- Loads the chosen dataset.  
- Splits into 80/20 (train/test).  
- Cleans + applies feature selection.  
- Saves:  
  - 'train_cleaned_<TECH>_<DATASET>.csv'  
  - 'test_cleaned_<TECH>_<DATASET>.cs'  

> Example: 'train_cleaned_MI_public.csv', 'test_cleaned_MI_public.csv' 



## 3.  Model Training  

Each model script (Random Forest, AdaBoost, Naïve Bayes) takes the cleaned train/test CSVs.  

### Approaches  
- **Within-dataset**  
  - Public → Public  
  - Private → Private  
- **Cross-dataset**  
  - Public → Private  
  - Private → Public  

> When cross-testing, the **test set columns are aligned to the train schema**, this is already handled in the code.  



## 4 - Outputs  

Model scripts save:  
- Metrics tables  
- Classification reports as png  
- Confusion matrices as png 
- ROC curves  as png
- Precision-Recall curves as png




