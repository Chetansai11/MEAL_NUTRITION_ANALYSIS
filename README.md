# 🥗 Meal Nutrition Analysis using Multimodal Deep Learning

This project presents a multimodal machine learning approach to estimate the **caloric and nutritional content of meals** by fusing data from diverse sources—**glucose sensors, food images, physical attributes, demographics, and gut microbiome profiles**. The model supports real-time, personalized nutrition prediction for digital health and wellness applications.

---

## 🎯 Objective

To develop a deep learning model that leverages cross-modal data to accurately predict meal nutrition values (primarily calories), enhancing dietary insights in personalized health tracking systems.

---

## 🧾 Data Modalities Used

| Source            | Type           | Description |
|------------------|----------------|-------------|
| 📈 CGM Data       | Time-Series    | Continuous glucose levels pre/post meal |
| 📷 Food Images    | Visual         | RGB images of meals processed via CNN |
| 👤 Demographics   | Tabular        | Age, gender, lifestyle habits |
| ⚖️ Physical Stats | Tabular        | Height, weight, BMI |
| 🧬 Microbiome      | Tabular        | Gut flora distribution via sequencing |

---

## 🧠 Model Architecture

A multimodal deep learning pipeline composed of:

- 🌀 **Convolutional Neural Networks (CNN)**  
  Extract visual features from food images for texture, size, and color analysis.

- ⏱️ **Long Short-Term Memory (LSTM)**  
  Process CGM time-series to capture physiological response to food intake.

- 🔗 **Fully Connected Layers**  
  Handle demographic, physical, and microbiome data, then fuse all features.

- 📦 **Fusion Layer**  
  Concatenates all learned representations for final regression prediction.

📘 **Architecture Diagram**  
![Model Diagram](https://github.com/Chetansai11/MEAL_NUTRITION_ANALYSIS/blob/main/model%20(1).png)

---

## 🏋️ Training & Evaluation

- **Loss Function:** Root Mean Squared Relative Error (RMSRE)
- **Optimizer:** Adam
- **Batch Size / Epochs:** Tuned via grid search
- **Regularization:** Dropout and early stopping
- **Augmentation:** Food image rotation, scaling, color jittering

📈 **Evaluation Metrics:**
- RMSRE (Root Mean Squared Relative Error)
- R² Score (Coefficient of Determination)
- Mean Absolute Error (MAE)

---

## 📊 Results

| Model Type           | Test RMSRE | Improvement Over Baseline |
|----------------------|------------|----------------------------|
| Baseline (CGM only)  | 52%        | —                          |
| Multimodal (CNN+LSTM+FC) | **33%** | **↓ 19%**                  |

- Achieved **34% improvement in accuracy** over single-modality baselines.
- Demonstrated robustness across varying user profiles and glucose behaviors.

---

## 🚀 Highlights

- 🔄 **Multimodal Integration**: Combines vision, time-series, and structured data into a unified predictive model.
- 📉 **Calorie Estimation**: Enables personalized nutrition tracking with medical-grade accuracy.
- 🧠 **Domain-aware Design**: Reflects real-world physiological, behavioral, and biological complexity.

---

## 🛠️ Tech Stack

- **Language**: Python  
- **Deep Learning Frameworks**: PyTorch (primary), Torchvision  
- **Model Architectures**:  
  - 🌀 **Convolutional Neural Networks (CNN)** for food image processing  
  - ⏱️ **Long Short-Term Memory (LSTM)** for CGM time-series analysis  
  - 🔗 **Fully Connected Layers** for demographic and microbiome data fusion  

- **Data Processing & Analysis**:  
  pandas, NumPy, scikit-learn, seaborn, Matplotlib  

- **Computer Vision Tools**:  
  OpenCV, Pillow (PIL)

- **Experimentation & Optimization**:  
  Grid Search, Early Stopping, Custom Loss Functions (RMSRE), Data Augmentation

- **Visualization**:  
  Matplotlib, Seaborn, TensorBoard (optional for training curves)

- **Environment**:  
  Jupyter Notebook, Google Colab, VS Code, Anaconda

---

## 🧪 Potential Extensions

- 🔍 **Nutrient Breakdown:** Extend predictions to include macronutrients (carbs, fats, proteins) and micronutrients (vitamins, minerals) beyond calorie estimation.
- 🕵️‍♀️ **Model Explainability:** Incorporate interpretability tools like **SHAP** or **LIME** to understand feature contributions across modalities.
- ⛅ **Cloud Deployment:** Deploy as a real-time API using **AWS**, **Render**, or **Streamlit Cloud** for integration into mobile health platforms.
- 🧪 **Personalized Diet Optimization:** Use microbiome feedback and demographic embeddings to recommend personalized meals.

---

## 📬 Contact

**Chetan Sai Borra**  
📧 sai311235@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/chetan-sai-16a252251/)

> *This project contributes toward real-world applications in digital nutrition, personalized medicine, and AI-assisted wellness systems.*
