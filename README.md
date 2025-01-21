# Meal Nutrition Analysis using Multimodal Learning

## Project Overview

The **Meal Nutrition Analysis** project aims to predict the nutritional value of meals by using multimodal learning techniques. The goal is to integrate different data sources, such as Continuous Glucose Monitors (CGM), food images, demographics, physical attributes, and gut microbiome data, to build a predictive model that estimates the calorie intake and other nutritional values of a meal.

The model uses advanced deep learning techniques, combining architectures such as Long Short-Term Memory (LSTM), Convolutional Neural Networks (CNN), and fully connected layers to process and analyze data from these different modalities. The project demonstrates the application of multimodal machine learning for real-world health monitoring and calorie prediction.

## Data Sources

The data sources used in this project include:
- **Continuous Glucose Monitors (CGM)**: Provides real-time glucose level data related to meal intake.
- **Food Images**: Images of the food items consumed, which are processed using CNN for feature extraction.
- **Demographics**: Includes age, gender, and other demographic information.
- **Physical Attributes**: Includes data like height, weight, and body mass index (BMI).
- **Gut Microbiome**: Data from microbiome sequencing, which influences metabolism and nutrition.

## Model Architecture

The model architecture is designed to handle the multimodal nature of the data. It integrates information from various sources using the following steps:
1. **Food Image Data**: Processed using CNNs for feature extraction.
2. **CGM Data**: Processed using LSTMs to capture temporal patterns in glucose levels.
3. **Demographic and Physical Attributes Data**: Combined through fully connected layers to enrich the model's understanding.

The multimodal features are fused together to make predictions about meal calorie intake, helping to build a comprehensive understanding of the meal's nutritional profile.

## Block Diagram

Below is the block diagram representing the architecture of the model:(model (1).png)

## Training and Evaluation

The model is trained on a labeled dataset of meals with known nutritional values, using a combination of loss functions tailored for regression and classification tasks. The performance is evaluated using metrics such as Mean Absolute Error (MAE) and R-squared (R²) to assess the model's accuracy in predicting calorie intake and other nutrition metrics.

## Results

The multimodal model outperforms baseline single-modality models by leveraging the complementary information from the different data sources. The model demonstrates a 30% improvement in accuracy for lunch calorie intake predictions compared to previous benchmarks.

This **README.md** provides an overview of the project, explains the architecture with a block diagram, and results.
