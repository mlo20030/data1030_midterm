# Food Categorization Using Machine Learning  

## Overview  
This project focuses on building and evaluating machine learning models to predict food categories based on their nutritional content. By leveraging various supervised learning algorithms, including Logistic Regression, Random Forest, K-Nearest Neighbors (KNN), and XGBoost, we aim to classify food items into one of five categories: Fruits, Vegetables, Dairy, Grains, and Meat. The project demonstrates the end-to-end machine learning pipeline, including data preprocessing, feature engineering, hyperparameter tuning, model evaluation, and interpretability using global feature importance.   

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
- **Micronutrients**   

---

## Results  
- The XGBoost model achieved the highest test accuracy of **92.59%**, outperforming the baseline accuracy of 20%.  
- Logistic Regression achieved an accuracy of **81.48%**, while Random Forest scored **83.95%**, and KNN achieved **81.48%**.  
- **Global Feature Importance** revealed that key predictors of food category included **Selenium, Vitamin B3, and Cholesterol**.  
- The precision-recall curves showed tradeoffs in model performance, with XGBoost demonstrating strong precision and recall.  


---

## File Descriptions  
| **File/Folder**      | **Description**                                         |
|---------------------|--------------------------------------------------------- |
| `data/`              | Contains raw dataset                                    |
| `figures/`           | Plots for EDA, model performance, and feature importance|
| `results/`           | Results files such as model predictions and evaluation  |
| `report/`            | Final report in PDF format                              |
| `src/`               | Source code and notebooks for preprocessing, modeling   |
| `.gitignore`         | Specifies files to ignore when pushing to GitHub        |
| `LICENSE`            | License file for the project                            |
| `README.md`          | This readme file                                        |
| `environment.yaml`   | YAML file specifying Python environment and dependencies|

---

## Key Findings  
- **Best Performing Model**: XGBoost achieved a test accuracy of **92.59%**, outperforming Logistic Regression, Random Forest, and KNN.  
- **Global Feature Importance**: Selenium, Vitamin B3, and Cholesterol were the most important features across the models.  

---

## Potential Improvements  
- **Data Enrichment**: Add features like cuisine type or preparation method to improve context.  
- **Handle Class Imbalance**: Apply oversampling or class-weighted models to balance underrepresented categories.  
- **Alternate Models**: Explore advanced models such as deep learning for more complex feature interactions.  
- **Interpretability**: Utilize SHAP for other models beyond XGBoost to better understand local feature effects.  

---

## References  
- Scikit-Learn Documentation: https://scikit-learn.org/  
- XGBoost Documentation: https://xgboost.readthedocs.io/  
- SHAP Documentation: https://shap.readthedocs.io/  
- Data Source: https://www.kaggle.com/datasets/beridzeg45/food-nutritional-facts/  

---

## License  
This project is licensed under the MIT License - see the `LICENSE` file for details.  

---

## Contact  
For questions, please contact **Morgan_Lo@brown.edu** or open an issue on the GitHub repository.  
