# Machine Failure Detection Using Generative AI: CMMS Approach

# Project Overview

This project focuses on improving predictive maintenance within Computerised Maintenance Management Systems (CMMS) using deep learning techniques and generative AI. The goal is to detect machine failures early, reduce unplanned downtime, and improve overall operational efficiency. By leveraging GANs (Generative Adversarial Networks), synthetic data is generated to address the class imbalance in the original dataset, leading to a more accurate failure detection model.

1. Table of Contents
2. Project Motivation
3. Project Flow
4. Dataset
5. Technologies Used
6. Modeling Approach
7. Results
8. How to Use
9. Next Steps

# Project Motivation
Predictive maintenance in industries often faces the challenge of limited failure data, making machine learning models prone to bias and underfitting. By implementing GANs, this project introduces synthetic data to augment the training set, improving the model’s ability to predict failures and reducing costly unplanned maintenance.

# Project Flow
![image](https://github.com/user-attachments/assets/48105392-1547-4f6f-b17f-e7f7200d8ced)

# Dataset
The dataset is based on historical machine performance data extracted from a CMMS. It contains both structured and unstructured data points related to:

Machine health indicators
Failure types (class labels)
Time-series performance metrics
To tackle the class imbalance between normal operations and failures, GANs were employed to generate synthetic failure instances, thereby improving the dataset balance.

Data gathered via GitHub https://github.com/shadgriffin/machine_failure

