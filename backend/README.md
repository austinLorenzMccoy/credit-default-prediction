<div align="center">
  <h1>🚀 CreditIntel Pro Backend</h1>
  <p><em>High-performance FastAPI backend for credit risk assessment and ML predictions</em></p>
  <img src="https://img.shields.io/badge/FastAPI-0.95.0-green" alt="FastAPI">
  <img src="https://img.shields.io/badge/Python-3.8+-blue" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.15.0-orange" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Pydantic-2.0.0-red" alt="Pydantic">
  <img src="https://img.shields.io/badge/Uvicorn-0.22.0-purple" alt="Uvicorn">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</div>

## 🎯 Overview

CreditIntel Pro Backend is a sophisticated, production-ready FastAPI application that provides machine learning-powered credit risk assessment and prediction services. Built with Python, TensorFlow, and modern async patterns, it delivers high-performance predictions with comprehensive error handling and monitoring.

### ✨ Key Features

- **⚡ High Performance**: Async FastAPI with uvicorn for concurrent request handling
- **🧠 ML Integration**: TensorFlow models for credit default and limit predictions
- **🔒 Type Safety**: Full Pydantic validation for request/response schemas
- **📊 Mock Predictions**: Intelligent mock predictor for development and testing
- **🔍 Comprehensive Logging**: Structured logging with configurable levels
- **🛡️ Error Handling**: Robust error handling with meaningful HTTP responses
- **📈 Health Monitoring**: Built-in health checks and system status endpoints
- **🔧 Modular Architecture**: Clean, maintainable code structure with separation of concerns

---

## 🛠️ Tech Stack

| Technology | Version | Description |
|------------|---------|-------------|
| **FastAPI** | 0.95.0+ | Modern, fast web framework for building APIs |
| **Python** | 3.8+ | Core programming language |
| **TensorFlow** | 2.15.0+ | Machine learning framework |
| **Pydantic** | 2.0.0+ | Data validation using Python type annotations |
| **Uvicorn** | 0.22.0+ | ASGI server for running FastAPI |
| **Scikit-learn** | Latest | Machine learning utilities and preprocessing |
| **Pandas** | Latest | Data manipulation and analysis |
| **NumPy** | Latest | Numerical computing |

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.8 or higher
- **pip** package manager
- **Virtual environment** (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/austinLorenzMccoy/credit-default-prediction.git
   cd credit-default-prediction/backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the API server:**
   ```bash
   python main.py
   ```

5. **Access the API:**
   - API Base URL: `http://localhost:8000`
   - Interactive Docs: `http://localhost:8000/docs`
   - Health Check: `http://localhost:8000/api/v1/health`

---

## 📁 Project Structure

```
backend/
├── app/                    # Main application code
│   ├── api/               # API routes and endpoints
│   │   ├── app.py         # FastAPI application factory
│   │   └── router.py      # API route definitions
│   ├── config/            # Configuration settings
│   │   ├── __init__.py
│   │   └── settings.py    # Application configuration
│   ├── models/            # ML model implementations
│   │   ├── __init__.py
│   │   ├── mock_predictor.py  # Mock predictor for development
│   │   ├── predictor.py   # Main ML predictor
│   │   └── train.py       # Model training utilities
│   ├── schemas/           # Pydantic request/response models
│   │   ├── __init__.py
│   │   └── request_models.py  # API data schemas
│   ├── utils/             # Utility functions
│   │   ├── __init__.py
│   │   ├── data_processor.py  # Data preprocessing utilities
│   │   └── model_loader.py     # Model loading utilities
│   └── __init__.py
├── scripts/               # CLI and utility scripts
│   ├── cli.py             # Command-line interface
│   └── run.py             # Application runner
├── tests/                 # Backend unit tests
│   └── test_model_loading.py  # Model loading tests
├── models/                # Pre-trained model files
│   ├── classification_model.keras
│   ├── regression_model.keras
│   ├── classification_scaler.pkl
│   └── regression_scaler.pkl
├── requirements.txt       # Python dependencies
├── main.py               # Application entry point
└── README.md             # This file
```

---

## 🔌 API Documentation

### Base URL
```
http://localhost:8000/api/v1
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_status": {
    "predictor_ready": true
  }
}
```

#### 2. Predict Credit Default Risk
```http
POST /predict/default
```

**Request Body:**
```json
{
  "credit_limit": 100000,
  "age": 25,
  "gender": 2,
  "education": 2,
  "marital_status": 1,
  "payment_status": 0,
  "bill_amount": 10000,
  "payment_amount": 2000
}
```

**Response:**
```json
{
  "prediction": "Low Risk of Default (Probability: 15.00%)",
  "probability": 0.15,
  "is_high_risk": false,
  "risk_factors": ["Low payment to bill ratio"]
}
```

#### 3. Predict Credit Limit
```http
POST /predict/credit-limit
```

**Request Body:** Same as default prediction

**Response:**
```json
{
  "predicted_credit_limit": 150000.0,
  "adjustment_factor": 1.5,
  "recommendation_factors": [
    "Good payment history",
    "High payment to bill ratio",
    "Age factor positive",
    "Education level positive"
  ]
}
```

### Request Parameters

| Parameter | Type | Description | Values |
|-----------|------|-------------|--------|
| `credit_limit` | number | Current credit limit | Numeric value |
| `age` | integer | Customer age | 18-100 |
| `gender` | integer | Customer gender | 1=male, 2=female |
| `education` | integer | Education level | 1=graduate, 2=university, 3=high school, 4=others |
| `marital_status` | integer | Marital status | 1=married, 2=single, 3=others |
| `payment_status` | integer | Last month's payment status | 0=paid duly, 1=1 month delay, 2=2 months delay, etc. |
| `bill_amount` | number | Last month's bill amount | Numeric value |
| `payment_amount` | number | Last month's payment amount | Numeric value |

---

## 🧠 Machine Learning Models

### Model Architecture

The backend supports two types of predictions:

1. **Default Risk Classification**
   - **Model**: Deep Neural Network with TensorFlow/Keras
   - **Input**: 23 financial features
   - **Output**: Default probability (0-1)
   - **Features**: Credit history, payment patterns, demographic data

2. **Credit Limit Regression**
   - **Model**: Neural Network regression model
   - **Input**: Same 23 features
   - **Output**: Recommended credit limit
   - **Logic**: Based on risk profile and payment behavior

### Mock Predictor

For development and testing, the backend uses `MockPredictor` which provides:

- **Intelligent Risk Scoring**: Based on payment history, bill ratios, and demographics
- **Realistic Credit Limits**: Calculated using adjustment factors
- **Explainable Factors**: Clear reasoning for predictions
- **Consistent Behavior**: Reproducible results for testing

---

## ⚙️ Configuration

### Settings

Configuration is managed through `app/config/settings.py`:

```python
# API Settings
API_TITLE = "Credit Default Prediction API"
API_VERSION = "1.0.0"
API_PREFIX = "/api/v1"

# Model Paths
CLASSIFICATION_MODEL_PATH = "models/classification_model.keras"
REGRESSION_MODEL_PATH = "models/regression_model.keras"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Environment Variables

You can override settings using environment variables:

```bash
export API_PORT=8000
export LOG_LEVEL=DEBUG
export MODEL_PATH=/path/to/models
```

---

## 🔧 Development

### Running the Server

**Development Mode:**
```bash
python main.py
# or using CLI
python scripts/cli.py api
```

**Custom Port:**
```bash
python scripts/cli.py api --port 8001
```

**Without Auto-reload:**
```bash
python scripts/cli.py api --no-reload
```

### Training Models

```bash
python scripts/cli.py train --data path/to/data.csv
```

### Running Tests

```bash
python scripts/cli.py test
# or using pytest
pytest backend/tests/
```

### Code Quality

```bash
# Type checking
mypy app/

# Code formatting
black app/

# Import sorting
isort app/

# Linting
flake8 app/
```

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t credit-default-prediction .
```

### Run Container

```bash
docker run -p 8000:8000 credit-default-prediction
```

### Docker Compose

```yaml
version: '3.8'
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
```

---

## 📊 Performance & Monitoring

### Logging

The application uses structured logging with multiple levels:

- **INFO**: General operational information
- **DEBUG**: Detailed debugging information
- **WARNING**: Warning messages
- **ERROR**: Error conditions

### Health Monitoring

- **Health Endpoint**: `/api/v1/health`
- **Model Status**: Checks if models are loaded and ready
- **System Metrics**: Request timing and error rates

### Performance Optimization

- **Async Processing**: Non-blocking request handling
- **Connection Pooling**: Efficient database connections
- **Caching**: Model loading optimization
- **Memory Management**: Efficient resource usage

---

## 🔒 Security

### Input Validation

- **Pydantic Schemas**: Type-safe request validation
- **Input Sanitization**: Protection against injection attacks
- **Rate Limiting**: Configurable request rate limits (future)

### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Best Practices

- **Environment Variables**: Sensitive data in environment
- **HTTPS**: Use TLS in production
- **Authentication**: Add JWT/OAuth for production (future)

---

## 🚀 Production Deployment

### Environment Setup

1. **Production Server:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

2. **Behind Reverse Proxy:**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. **Process Management:**
   ```bash
   # Using systemd
   sudo systemctl start credit-api
   sudo systemctl enable credit-api
   ```

### Monitoring

- **Application Logs**: Structured JSON logging
- **Health Checks**: Automated health monitoring
- **Metrics**: Prometheus integration (future)
- **Alerting**: Error rate and performance alerts

---

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Verify model files exist in `models/` directory
   - Check file permissions
   - Ensure correct TensorFlow version

2. **Port Conflicts**
   - Check if port is in use: `lsof -i :8000`
   - Use different port: `python main.py --port 8001`

3. **Import Errors**
   - Activate virtual environment
   - Install dependencies: `pip install -r requirements.txt`
   - Check Python version compatibility

4. **CORS Issues**
   - Configure allowed origins in production
   - Check frontend API URL configuration

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python main.py
```

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Guidelines

- Follow **PEP 8** style guidelines
- Add **type hints** for all functions
- Write **comprehensive tests** for new features
- Update **documentation** for API changes
- Use **descriptive commit messages**

---

## 📄 License

This project is licensed under the MIT License - see the main project LICENSE file for details.

---

## 🙏 Acknowledgments

- **FastAPI Team** - For the amazing web framework
- **TensorFlow Team** - For the ML framework
- **Pydantic Team** - For data validation tools
- **Python Community** - For the incredible ecosystem

---

## 📞 Support

For support and questions:

- 📧 **Email**: chibuezeaugustine23@gmail.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/austinLorenzMccoy/credit-default-prediction/issues)
- 📖 **Documentation**: [Project Wiki](https://github.com/austinLorenzMccoy/credit-default-prediction/wiki)
- 🔗 **API Docs**: Available at `/docs` endpoint when running

---

<div align="center">
  <p>🚀 Built for high-performance financial risk assessment</p>
  <p>© 2024 CreditIntel Pro. All rights reserved.</p>
</div>
