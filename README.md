# 🌟 Credit Prediction Machine Learning API  

## 🚀 Project Overview  
This project is a state-of-the-art **Machine Learning API** designed to assist financial institutions with credit risk assessment. It leverages advanced neural networks for:  
- **Credit Card Default Prediction**: Assess the probability of a customer defaulting on their credit card payments.  
- **Credit Limit Recommendation**: Estimate an optimal credit limit based on customer financial behavior.  

With a clean architecture and scalable design, this API bridges the gap between machine learning models and practical financial solutions.  

---

## 🔧 Tech Stack  
| **Category**          | **Technology**                    |  
|-----------------------|-----------------------------------|  
| Programming Language  | Python                            |  
| Machine Learning      | TensorFlow/Keras, Scikit-learn    |  
| Data Manipulation     | Pandas, NumPy                     |  
| API Framework         | FastAPI                           |  
| Data Scaling          | StandardScaler, Random Oversampling |  
---

## 📊 Features  

### **Default Prediction Model**  
- **Binary Classification**: Predicts the likelihood of credit card default.  
- **Probability Output**: Provides interpretable percentage risk scores.  
- **Class Balancing**: Handles imbalanced datasets using **Random Oversampling**.  

### **Credit Limit Prediction Model**  
- **Regression-based**: Estimates appropriate credit limits for customers.  
- **Regularization**: Incorporates techniques to avoid overfitting.  
- **Custom Feature Engineering**: Tailored financial predictors for better performance.  

---

## 🛠️ Core Capabilities  

### Machine Learning Highlights  
- **Deep Neural Networks**: Custom layers with dropout and regularization.  
- **Feature Scaling**: Scaled inputs ensure consistent model performance.  
- **Model Evaluation**: Metrics like Mean Squared Error and Classification Accuracy guide training.  

### API Development  
- **FastAPI Framework**: Efficient, asynchronous API handling.  
- **Pydantic for Validation**: Validates incoming request payloads.  
- **Robust Error Handling**: Ensures predictable and stable responses.  

---

## 🚦 Getting Started  

### Prerequisites  
Ensure you have the following installed:  
- Python 3.8+  
- pip (Python package manager)  
- Virtual environment (recommended)  

### Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/austinLorenzMccoy/credit-prediction-api  
   cd credit-prediction-api  
   ```  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

### Running the API  
To start the API server locally:  
```bash  
python cli.py api  
```  

## 📡 API Endpoints  

### 1️⃣ **Default Prediction**  
- **URL**: `/api/v1/predict/default`  
- **Method**: `POST`  
- **Input**: Simplified customer financial data  
- **Output**: Default risk as a percentage probability with risk factors  

### 2️⃣ **Credit Limit Prediction**  
- **URL**: `/api/v1/predict/credit-limit`  
- **Method**: `POST`  
- **Input**: Simplified customer financial data  
- **Output**: Recommended credit limit with adjustment factors  

### 3️⃣ **Health Check**  
- **URL**: `/api/v1/health`  
- **Method**: `GET`  
- **Output**: API health status and model status  

## 🔍 Example API Usage with curl

### Check API Health
```bash
curl -X GET http://localhost:8000/api/v1/health
```

Response:
```json
{"status":"healthy","model_status":{"predictor_ready":true}}
```

### Predict Default Risk
```bash
curl -X POST http://localhost:8000/api/v1/predict/default \
  -H "Content-Type: application/json" \
  -d '{
    "credit_limit": 100000,
    "age": 25,
    "gender": 2,
    "education": 2,
    "marital_status": 1,
    "payment_status": 0,
    "bill_amount": 10000,
    "payment_amount": 2000
  }'
```

Response:
```json
{"prediction":"Low Risk of Default (Probability: 15.00%)","probability":0.15,"is_high_risk":false,"risk_factors":["Low payment to bill ratio"]}
```

### Predict Credit Limit
```bash
curl -X POST http://localhost:8000/api/v1/predict/credit-limit \
  -H "Content-Type: application/json" \
  -d '{
    "credit_limit": 100000,
    "age": 35,
    "gender": 2,
    "education": 1,
    "marital_status": 1,
    "payment_status": 0,
    "bill_amount": 10000,
    "payment_amount": 8000
  }'
```

Response:
```json
{"predicted_credit_limit":150000.0,"adjustment_factor":1.5,"recommendation_factors":["Good payment history","High payment to bill ratio","Age factor positive","Education level positive"]}
```

## 📊 Input Parameters Explained

| Parameter | Description | Values |
|-----------|-------------|--------|
| `credit_limit` | Current credit limit | Numeric value |
| `age` | Customer age | Integer |
| `gender` | Customer gender | 1=male, 2=female |
| `education` | Education level | 1=graduate, 2=university, 3=high school, 4=others |
| `marital_status` | Marital status | 1=married, 2=single, 3=others |
| `payment_status` | Last month's payment status | 0=paid duly, 1=1 month delay, 2=2 months delay, etc. |
| `bill_amount` | Last month's bill amount | Numeric value |
| `payment_amount` | Last month's payment amount | Numeric value |

---

## 📡 API Endpoints  

### 1️⃣ **Default Prediction**  
- **URL**: `/predict/default`  
- **Method**: `POST`  
- **Input**: Customer's financial features (list of floats)  
- **Output**: Default risk as a percentage probability  

### 2️⃣ **Credit Limit Prediction**  
- **URL**: `/predict/credit-limit`  
- **Method**: `POST`  
- **Input**: Customer's age, payment history, and bill amount  
- **Output**: Recommended credit limit (numeric)  

---

## 📈 Model Performance  

### Default Prediction Model  
- **Accuracy**: Validated on a balanced dataset with enhanced features.  

### Credit Limit Prediction Model  
- **Metric**: Mean Squared Error minimized through iterative fine-tuning.  

---

## 📂 Project Structure  

```plaintext  
credit-prediction-api/  
├── app/  
│   ├── app.py             # FastAPI application  
│   ├── main.py            # Model training script  
│   ├── models/            # Pre-trained models  
│   ├── scalers/           # Saved scalers  
├── data/  
│   ├── raw/               # Original datasets  
├── notebooks/             # Jupyter notebooks for EDA and model training  
├── frontend/              # HTML, CSS, and JS files for the user interface  
├── tests/                 # Unit tests for endpoints  
├── requirements.txt       # Python dependencies  
├── README.md              # Project documentation  
└── .env                   # Environment variables  
```  

---

## 🔒 Data Privacy  
- Models trained on anonymized data.  
- No sensitive information is stored or processed beyond the scope of prediction tasks.  

---

## 🤝 Contributing  

We welcome contributions! To get started:  
1. Fork this repository.  
2. Create a feature branch: `git checkout -b feature-name`.  
3. Commit your changes: `git commit -m "Add new feature"`.  
4. Push the branch: `git push origin feature-name`.  
5. Open a pull request for review.  

---

## 📄 License  

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.  

---

## 👥 Contact  

- **GitHub**: [@austinLorenzMccoy](https://github.com/austinLorenzMccoy)  
- **Email**: chibuezeaugustine23@gmail.com 

