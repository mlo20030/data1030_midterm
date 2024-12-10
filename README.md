# Food Categorization Using Machine Learning  

## Overview  
This project focuses on building and evaluating machine learning models to predict food categories based on their nutritional content. By leveraging various supervised learning algorithms, including Logistic Regression, Random Forest, K-Nearest Neighbors (KNN), and XGBoost, we aim to classify food items into one of five categories: Fruits, Vegetables, Dairy, Grains, and Meat. The project demonstrates the end-to-end machine learning pipeline, including data preprocessing, feature engineering, hyperparameter tuning, model evaluation, and interpretability using global and local feature importance.  

The final deliverables for this project include a technical report, presentation slides, and this well-organized repository. The report discusses the problem, methods, evaluation metrics, model results, and potential areas for improvement.  

---

## Project Structure  
```
.
├── data/              # Raw and preprocessed data files
├── figures/           # Plots, confusion matrices, and visual results
├── results/           # Model predictions, saved models, and evaluation metrics
├── report/            # PDF version of the final report
├── src/               # Source code (Jupyter Notebooks, Python scripts, etc.)
├── .gitignore         # Git ignore file
├── LICENSE            # License file for the project
└── README.md          # This file
```

---

## Dataset  
The dataset used in this project is the Foodstruct Nutritional Facts dataset, which includes information on the nutritional content of various food items. Each row corresponds to a specific food, and columns represent its nutritional content, such as calories, protein, fats, carbohydrates, and other micronutrients. The target variable is the **Category Name**, which classifies each food item into one of five categories: **Fruits, Vegetables, Dairy, Grains, and Meat**.  

Key features include:  
- **Calories**  
- **Protein**  
- **Fats**  
- **Carbohydrates**  
- **Micronutrients** (e.g., Selenium, Zinc, and various vitamins)  

---

## Machine Learning Pipeline  
1. **Data Preprocessing**  
   - Handling missing data:  
     - **XGBoost**: Built-in handling of missing continuous variables.  
     - **Other Models (LR, RF, KNN)**: Reduced features approach where columns with missing values are dropped.  
   - Data normalization and scaling: Applied to numeric features for Logistic Regression, Random Forest, and KNN.  

2. **Modeling**  
   - Models used:  
     - Logistic Regression (Baseline model)  
     - Random Forest  
     - K-Nearest Neighbors (KNN)  
     - XGBoost  
   - Hyperparameter tuning was conducted using **GridSearchCV** with 5-fold cross-validation.  
   - The best hyperparameters were selected for each model.  

3. **Evaluation**  
   - Baseline accuracy (20%) calculated as the proportion of the most frequent class.  
   - Performance metrics:  
     - Accuracy  
     - Confusion matrices for each model  
     - Global and local feature importance (SHAP values for XGBoost)  
     - Precision-Recall curves to evaluate precision/recall tradeoffs  

4. **Model Interpretability**  
   - **Global Feature Importance**: Feature importance rankings were displayed for Random Forest and XGBoost.  
   - **Local Feature Importance (SHAP Values)**: SHAP values were calculated to understand how specific features influenced individual predictions.  

---

## Results  
- The XGBoost model achieved the highest test accuracy of **92.59%**, outperforming the baseline accuracy of 20%.  
- Logistic Regression achieved an accuracy of **81.48%**, while Random Forest scored **83.95%**, and KNN achieved **81.48%**.  
- **Global Feature Importance** revealed that key predictors of food category included **Selenium, Vitamin B3, and Cholesterol**.  
- The precision-recall curves showed tradeoffs in model performance, with XGBoost demonstrating strong precision and recall.  

---

## Installation and Reproducibility  
### **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/food-categorization-ml.git
cd food-categorization-ml
```

### **2. Set Up a Virtual Environment**  
To ensure the project runs smoothly, it is recommended to create a virtual environment.  
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

### **3. Install Required Packages**  
Install the required dependencies from the `environment.yaml` file.  
```bash
conda env create -f environment.yaml
conda activate food-categorization
```

Alternatively, you can use `requirements.txt` if using `pip`:  
```bash
pip install -r requirements.txt
```

---

**## Dependencies**  
This project requires the following libraries and dependencies:  
| **Package**         | **Version**    |
|--------------------|----------------|
| Python             | 3.12           |
| Pandas             | >=1.5.0        |
| NumPy              | >=1.21.0       |
| Scikit-Learn       | >=1.2.0        |
| Matplotlib         | >=3.6.0        |
| Seaborn            | >=0.11.0       |
| XGBoost            | >=1.6.0        |
| SHAP               | >=0.39.0       |

---

**## Usage Instructions**  
To reproduce the results, follow these steps:  
1. **Run Data Preprocessing**: Run the Jupyter notebook located in `src/data_preprocessing.ipynb` to prepare the dataset.  
2. **Train Models**: Run the `src/model_training.ipynb` notebook to train Logistic Regression, Random Forest, XGBoost, and KNN models.  
3. **Evaluate Models**: Evaluate model performance using confusion matrices, precision-recall curves, and feature importance plots.  
4. **Generate Report**: Export the final report as a PDF and submit it via Gradescope.  

---

**## File Descriptions**  
| **File/Folder**      | **Description**                                           |
|---------------------|---------------------------------------------------------|
| `data/`              | Contains raw and preprocessed datasets                    |
| `figures/`           | Plots for EDA, model performance, and feature importance |
| `results/`           | Results files such as model predictions and evaluation   |
| `report/`            | Final report in PDF format                              |
| `src/`               | Source code and notebooks for preprocessing, modeling    |
| `.gitignore`         | Specifies files to ignore when pushing to GitHub        |
| `LICENSE`            | License file for the project                            |
| `README.md`          | This readme file                                        |
| `environment.yaml`   | YAML file specifying Python environment and dependencies |
| `requirements.txt`   | Alternative file for pip-based dependency installation   |

---

**## Key Findings**  
- **Best Performing Model**: XGBoost achieved a test accuracy of **92.59%**, outperforming Logistic Regression, Random Forest, and KNN.  
- **Global Feature Importance**: Selenium, Vitamin B3, and Cholesterol were the most important features across the models.  
- **SHAP Values**: SHAP analysis revealed how individual features influenced predictions, offering local interpretability for the XGBoost model.  

---

**## Potential Improvements**  
- **Data Enrichment**: Add features like cuisine type or preparation method to improve context.  
- **Handle Class Imbalance**: Apply oversampling or class-weighted models to balance underrepresented categories.  
- **Alternate Models**: Explore advanced models such as deep learning for more complex feature interactions.  
- **Interpretability**: Utilize SHAP for other models beyond XGBoost to better understand local feature effects.  

---

**## References**  
- Scikit-Learn Documentation: https://scikit-learn.org/  
- XGBoost Documentation: https://xgboost.readthedocs.io/  
- SHAP Documentation: https://shap.readthedocs.io/  
- Data Source: https://www.kaggle.com/datasets/beridzeg45/food-nutritional-facts/  

---

**## Contributing**  
Contributions are welcome! Please fork the repository and submit a pull request for any improvements.  

---

**## License**  
This project is licensed under the MIT License - see the `LICENSE` file for details.  

---

**## Contact**  
For questions, please contact **Morgan_Lo@brown.edu** or open an issue on the GitHub repository.  
