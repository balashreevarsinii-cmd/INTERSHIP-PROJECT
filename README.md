# INTERSHIP-PROJECT

**Projects Undertaken During Internship**

**Project 1: Sentiment Analysis for Product Reviews using Machine Learning and NLP**

**Project Overview:**

This project focused on building an intelligent sentiment analysis system to automatically classify customer product reviews as Positive, Negative, or Neutral. The aim was to understand customer opinions at scale and help businesses improve products and services based on feedback.

**Work Description:**

Collected and worked with a large-scale Amazon Consumer Reviews dataset containing thousands of real-world product reviews.

Performed data cleaning and preprocessing, including removal of punctuation, stop-words, special characters, and conversion of text to lowercase.

Applied tokenization and lemmatization techniques to normalize the textual data and improve model accuracy.

Converted raw text into numerical features using TF-IDF (Term Frequency–Inverse Document Frequency) vectorization.

Designed and trained a Multinomial Naïve Bayes classifier to perform sentiment classification.

Split the dataset into training and testing sets to ensure proper validation of the model.

Evaluated the model using accuracy, precision, recall, and F1-score, achieving high classification performance (~91% accuracy).

**Advanced Implementation:**

Integrated an NLP-based generative AI model (Gemini API) to provide context-aware sentiment reasoning in addition to traditional ML predictions.

Implemented real-time user input handling for both single and multiple reviews, generating structured JSON outputs containing sentiment, confidence, and reasoning.

**Outcome & Impact:**

Developed a hybrid ML + NLP system capable of both statistical classification and semantic reasoning.

The solution can be deployed for e-commerce review analysis, customer feedback monitoring, and business intelligence applications.

**Technologies Used:**
Python, Pandas, NLTK, Scikit-learn, TF-IDF, Multinomial Naïve Bayes, Generative AI (Gemini API)

**Project 2: Credit Card Fraud Detection Using Machine Learning**

**Project Overview:**
This project aimed to detect fraudulent credit card transactions in real time using supervised machine learning techniques, helping reduce financial losses and enhance transaction security.

**Work Description:**

Worked with a highly imbalanced credit card transaction dataset containing anonymized transaction features (V1–V28), transaction amount, and class labels.

Performed extensive exploratory data analysis (EDA) to understand fraud patterns and data imbalance.

Applied feature scaling and normalization to prepare numerical data for model training.

Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique) to improve fraud detection accuracy.

Implemented and trained an XGBoost Classifier to identify fraudulent transactions.

Evaluated model performance using confusion matrix, ROC-AUC score, precision, recall, and F1-score.

Built a probability-based decision system to flag transactions as Fraudulent or Normal.

**Outcome & Impact:**

Achieved highly reliable fraud detection with strong recall for minority (fraud) cases.

The system is suitable for banking systems, payment gateways, and financial risk monitoring platforms.

**Technologies Used:**
Python, Pandas, Scikit-learn, XGBoost, SMOTE, NumPy, Matplotlib

**Project 3: Customer Churn Prediction for Telecom Industry**

Project Overview:
The objective of this project was to predict whether a telecom customer is likely to discontinue services, enabling companies to take preventive retention actions.

**Work Description:**

Analyzed a telecom customer dataset containing demographic, service usage, billing, and contract information.

Handled missing values and performed data type corrections and encoding of categorical features.

Applied feature engineering to extract meaningful predictors related to churn behavior.

Used Gradient Boosting Classifier for predictive modeling.

Evaluated the model using accuracy and class-wise performance metrics.

Conducted feature importance analysis to identify key factors influencing customer churn (contract type, monthly charges, internet service).

**Outcome & Impact:**

Enabled early identification of customers at high risk of churn.

Provided actionable insights for customer retention strategies and business decision-making.

**Technologies Used:**
Python, Pandas, Scikit-learn, Gradient Boosting, Data Visualization

**Project 4: Retail Sales Forecasting Using Time-Series Analysis**

**Project Overview:**
This project focused on forecasting future retail sales using historical data to help businesses optimize inventory management and financial planning.

**Work Description:**

Worked with historical retail sales datasets containing date-wise and store-wise sales information.

Preprocessed time-series data by handling missing values and aggregating sales data.

Implemented Facebook Prophet, a robust time-series forecasting model, to capture trends and seasonality.

Evaluated forecasting accuracy using Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

Visualized future sales predictions and seasonal components.

**Outcome & Impact:**

Delivered accurate sales forecasts for future time periods.

Useful for inventory optimization, demand planning, and strategic retail decision-making.

**Technologies Used:**
Python, Pandas, Prophet, Matplotlib, Time-Series Analysis

**Independent Project: Iron Dome – AI-Based Real-Time Missile Interception Decision System**

**Project Nature:**
Self-Initiated / Independent Research & Development Project
(Designed, implemented, and documented independently; code hosted on GitHub)

**Project Overview:**
The Iron Dome Project is an advanced AI-driven defense simulation system developed to model and predict real-time missile threats and make intelligent interception decisions. The project integrates physics-based trajectory modeling, machine learning, and decision-making logic to distinguish between threat and non-threat missiles and recommend optimal interception strategies.

**Problem Statement:**
Traditional missile defense systems either rely heavily on physics-based rules or purely data-driven AI models. This project addresses the limitation by combining both approaches to achieve faster, safer, and more reliable interception decisions, especially under real-time constraints.

**Work Description:**

Designed a synthetic missile trajectory generation system using kinematic and projectile motion equations (position, velocity, angle, time).

Generated labeled datasets representing threat and non-threat missiles based on parameters such as speed, altitude, trajectory angle, and time-to-impact.

Developed and trained a Feedforward Neural Network (FNN) to classify incoming missiles in real time.

Implemented decision-logic rules to determine:

Whether interception is required

Which missile should be intercepted

When interception should be avoided to reduce collateral damage

Integrated AI predictions with physics constraints, ensuring that interception decisions are both mathematically valid and operationally safe.

Designed a simulation visualization to clearly show missile paths, interceptor paths, and classification outputs.

Built the project in a modular manner so that it can be extended to:

Edge-AI deployment

Real-time radar input

Multi-missile swarm scenarios

**Key Innovations:**

Hybrid approach combining Physics + AI, instead of relying on a single method

Real-time threat vs non-threat missile discrimination

Safety-aware interception logic to avoid unnecessary launches

Fully reproducible and documented system hosted on GitHub

**Outcome & Impact:**

Demonstrated how AI can support defense decision-making systems under time-critical scenarios.

The project serves as a strong foundation for defense simulations, edge-AI systems, and autonomous interception research.

Successfully showcased as a research-oriented and innovation-driven personal project, suitable for academic, hackathon, and publication purposes.

**Technologies Used:**
Python, Machine Learning (FNN), Numerical Simulation, Physics-Based Modeling, Data Visualization, GitHub Version Control

**Project Repository:**
Source code, documentation, and simulation results are maintained on GitHub as an original independent project.
