# 🐟 Fish Weight Predictor AI

A high-precision Machine Learning application designed to estimate fish mass based on morphological measurements. This project follows a professional **End-to-End Data Science Roadmap**, utilizing Log-Transformation to ensure physical consistency and eliminate negative weight predictions.

---

## 🛠️ Technical Architecture

### 1. Data Science Pipeline

* **Data Cleaning:** Systematic removal of redundant identifiers and standardization of categorical species data.
* **Outlier Management:** Implemented a 95th percentile clip on the target variable to stabilize the linear gradient.
* **Feature Engineering:**

  * Addressed **Multicollinearity** by consolidating 99% correlated length metrics into a single representative feature (`length3_cm`).
  * Applied **Log-Transformation** (y = ln(Weight)) to handle the non-linear relationship between volume and mass.
* **Scaling:** Z-score normalization using training set statistics to prevent data leakage.

---

### 2. Model Performance

The model was trained using **Multiple Linear Regression** on a diverse ichthyometric dataset:

* **R² Score:** 0.85+ 
* **Inference:** Utilizes exponential reconstruction (e^y) to ensure all predicted weights are strictly positive.

---

## 💻 Tech Stack

* **Language:** Python 3.11
* **ML Framework:** Scikit-Learn
* **Data Handling:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib
* **Deployment:** Streamlit (Cloud)
* **Serialization:** Pickle (Version-locked)

---

## 📁 Repository Structure

```
├── app.py                # Streamlit UI & Inference Logic
├── fish_model_elite.pkl  # Serialized Model, Scalers, and Metadata
├── requirements.txt      # Production Dependencies
└── README.md             # Project Documentation
```

---

## 🔧 Installation & Local Usage

1. **Clone the repository**

```bash
git clone https://github.com/abinayaav21-art/Fish-Weight-predictor.git
cd Fish-Weight-predictor
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the App**

```bash
streamlit run app.py
```

---

## 👤 Author

**Abinaya**
*Data Science Intern*
