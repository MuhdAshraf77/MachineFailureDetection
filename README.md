# Machine Failure Detection Using Generative AI: CMMS Approach

# Project Overview

This project focuses on improving predictive maintenance within Computerised Maintenance Management Systems (CMMS) using deep learning techniques and generative AI. The goal is to detect machine failures early, reduce unplanned downtime, and improve overall operational efficiency. By leveraging GANs (Generative Adversarial Networks), synthetic data is generated to address the class imbalance in the original dataset, leading to a more accurate failure detection model.

1. Table of Contents
2. Project Motivation
3. Dataset
4. Technologies Used
5. Project Flow
6. Modeling Approach
7. Results
8. Deployment

# Project Motivation
Predictive maintenance in industries often faces the challenge of limited failure data, making machine learning models prone to bias and underfitting. By implementing GANs, this project introduces synthetic data to augment the training set, improving the model’s ability to predict failures and reducing costly unplanned maintenance.

# Dataset
The dataset is based on historical machine performance data extracted from a CMMS. It contains both structured and unstructured data points related to:

Machine health indicators
Failure types (class labels)
Time-series performance metrics
To tackle the class imbalance between normal operations and failures, GANs were employed to generate synthetic failure instances, thereby improving the dataset balance.

Data gathered via GitHub https://github.com/shadgriffin/machine_failure

# Technologies Used
Languages: Python, SQL
Libraries: TensorFlow, Keras, Scikit-learn, Pandas, NumPy
Tools & Platforms: Google Cloud Platform (GCP), VSCode, GitHub
Modeling Frameworks: Generative Adversarial Networks (GANs), Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTM) networks, Recurrent Neural Networks (RNNs)
Other Tools: GCP for deployment and Open AI API Integration for recommendation insights

# Project Flow
![image](https://github.com/user-attachments/assets/48105392-1547-4f6f-b17f-e7f7200d8ced)

# Modeling Approach
1. Data Preprocessing:
- Cleaned and normalized the dataset.
- Handled missing values and outliers.
- Balanced the dataset using GANs and SMOTE to generate synthetic failure data.

2. Model Selection:
- Generative Model: GANs and SMOTE were used to address the class imbalance.
- Predictive Models: CNN, RNN and LSTM were used to capture both temporal and spatial patterns in the data.

3. Evaluation:
- K-fold Cross-Validation and Hyperparameter tuning were performed.
- The model performance was evaluated based on performance metrics and AUPRC (Area Under Precision-Recall Curve).

# Results

Comparative Analysis Summary

CNN Model Performance:
Excelled in the original dataset with high recall and reasonable AUROC, proving effective in scenarios where nearly all failures need to be identified, despite a higher false positive rate.
Demonstrated superior performance on the GANs dataset with 95.75% accuracy, 81% recall, and a ROC score of 0.97, making it ideal for deployment due to its strong ability to differentiate between failure and non-failure states while minimizing false alarms.

RNN Model Performance:
Best suited for the SMOTE dataset, balancing sensitivity and precision well with fewer false positives compared to the CNN on the GANs dataset, though it fell short in recall compared to CNN.

LSTM Model Performance:
The LSTM model showed competitive results, particularly in handling sequential data, but it did not surpass the CNN in terms of overall accuracy and recall on the GANs dataset, making CNN the preferred choice for deployment.

Model Selection for Deployment:
The CNN model, trained on the GANs dataset, was chosen for deployment. It offers a robust solution for predictive maintenance, effectively managing the trade-offs between detecting true failures (sensitivity) and avoiding false alarms (specificity).
This summary highlights the performance strengths and justifications for selecting the CNN model for real-world application, focusing on its superior handling of data from the GANs dataset.

# Deployment

After conducting comprehensive dataset analyses, the CNN model trained on GANs data was selected for deployment due to its superior performance. The deployment was carried out on Google Cloud Platform (GCP), which provides a robust and scalable infrastructure ideal for deploying advanced machine learning models.

Key Steps in the Deployment:

1. Model Saving & Versioning:
The trained model was saved in a private GitHub repository for version control and easy access.

2. Containerization:
The model was containerized using Docker, ensuring consistent deployment across different environments by packaging the application and its dependencies together.

3. Google Cloud Run:
For deployment, Google Cloud Run was utilized. It simplifies the process by automatically handling the logistics of serving containerized applications, removing the need to manually orchestrate the environment.

4. Automated Building & Deployment:
Google Cloud Build was used for automating the building and deploying processes of the container, making the deployment pipeline more efficient and less error-prone.

5. Integration with OpenAI's API:
The system was integrated with OpenAI’s API to enhance the model's functionality, enabling the generation of smarter insights and actionable recommendations based on machine failure predictions. This integration allows the system not only to forecast potential malfunctions but also to suggest measures to improve maintenance strategies in industrial applications.

6. Automated Failure Detection:
After uploading machine data, the system automatically detects potential failures. The output includes the number of machines predicted to fail and those expected to operate correctly, displayed in an intuitive table format for each machine.

![image](https://github.com/user-attachments/assets/7ef1a07a-5eb9-4eae-a724-0dc3b9d87f1b)

The process of detecting potential failures is done automatically by the system after uploading the machine data. It shows the counts of machines predicted to fail and those that are going to operate well. The results are shown on a table with a column for each machine making it easy to see the condition of each machine.

![image](https://github.com/user-attachments/assets/2b93553f-149e-451d-8932-172cbcce6ada)

In the individual value entry tab, users can input delimited feature values for a specific machine and immediately receive a prediction. This flexibility allows for rapid assessment without the need for navigating complex menus or interfaces.

![image](https://github.com/user-attachments/assets/fa490414-7b7d-4b6d-bb8f-bc0b1129857e)

If the prediction implies that no failure is going to happen soon, the system provides recommendations on how the machine should be operated for long-term reliability and efficiency with additional advice on preventive maintenance.

![image](https://github.com/user-attachments/assets/ac5b0801-69b6-494c-bdc8-a2271d9e8d80)

However, if a device is set to fail then in the same case the system gives personalized suggestions and insight on what maintenance or repair it needs. The user gains proactive advice on how to reduce risks and make predictable interventions that will help avoid unscheduled downtime of the machine.

![image](https://github.com/user-attachments/assets/588861cf-0938-4f1e-8cc3-fd5273f0f84f)


