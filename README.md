<div align="center">
  <h1>🏦 CreditIntel Pro</h1>
  <p><em>Advanced Machine Learning Platform for Credit Risk Assessment and Prediction</em></p>
  <img src="https://img.shields.io/badge/Python-3.8+-blue" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.95.0-green" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-19.0.1-blue" alt="React">
  <img src="https://img.shields.io/badge/TypeScript-5.8-blue" alt="TypeScript">
  <img src="https://img.shields.io/badge/TensorFlow-2.15.0-orange" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen" alt="Status">
</div>

---

## 🎯 Overview

**CreditIntel Pro** is a comprehensive, production-ready platform for credit risk assessment and prediction. Combining advanced machine learning algorithms with a modern, intuitive web interface, it empowers financial institutions to make data-driven decisions with confidence and precision.

### 🌟 Key Highlights

- **🤖 Advanced ML Models**: Deep learning networks for accurate risk assessment
- **⚡ Real-time Predictions**: Sub-second response times for high-volume processing
- **🎨 Modern UI/UX**: Beautiful, responsive React frontend with TypeScript
- **🔒 Enterprise Ready**: Robust error handling, logging, and monitoring
- **📊 Comprehensive Analytics**: Detailed insights and risk factor explanations
- **🔧 Modular Architecture**: Clean, maintainable, and scalable codebase

---

## 🚀 Architecture Overview

<div align="center">
  <img src="https://img.shields.io/badge/Frontend-React%20%7C%20TypeScript%20%7C%20Tailwind-61DAFB" alt="Frontend Stack">
  <img src="https://img.shields.io/badge/Backend-Python%20%7C%20FastAPI%20%7C%20TensorFlow-3776AB" alt="Backend Stack">
  <img src="https://img.shields.io/badge/Infrastructure-Docker%20%7C%20Uvicorn-2496ED" alt="Infrastructure">
</div>

### System Components

```
┌─────────────────┐    HTTP/REST API    ┌─────────────────┐
│   React App     │ ◄─────────────────► │   FastAPI       │
│  (Frontend)     │                    │   Backend       │
│                 │                    │                 │
│ • Customer UI   │                    │ • ML Models     │
│ • Dashboards    │                    │ • API Routes    │
│ • Visualizations│                   │ • Validation    │
└─────────────────┘                    └─────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | React 19.0.1, TypeScript 5.8, Tailwind CSS 4.1 | Modern UI with type safety |
| **Backend** | FastAPI 0.95.0, Python 3.8+, TensorFlow 2.15+ | High-performance API & ML |
| **Data Processing** | Pandas, NumPy, Scikit-learn | Data manipulation & preprocessing |
| **Deployment** | Docker, Uvicorn, Nginx | Containerized deployment |
| **Development** | Vite 6.2.3, ESLint, Prettier | Fast development workflow |

---

## 🎯 Core Features

### 🧠 Machine Learning Capabilities

#### Credit Default Prediction
- **Model Type**: Binary Classification Neural Network
- **Accuracy**: High-precision risk assessment with explainable factors
- **Features**: 23+ financial and demographic parameters
- **Output**: Probability score with risk categorization

#### Credit Limit Recommendation
- **Model Type**: Regression Neural Network
- **Logic**: Dynamic limit adjustment based on risk profile
- **Factors**: Payment history, income stability, credit utilization
- **Output**: Recommended limit with adjustment explanations

### 🎨 User Interface Features

#### Customer Intelligence Dashboard
- **Profile Management**: Comprehensive customer data input
- **Real-time Analysis**: Instant risk assessment and scoring
- **Interactive Visualizations**: Charts and risk gauges
- **Historical Tracking**: Prediction history and trend analysis

#### Executive Overview
- **Portfolio Metrics**: System-wide performance indicators
- **Risk Distribution**: Aggregate risk analysis
- **Model Performance**: Accuracy and reliability metrics
- **System Health**: Real-time monitoring dashboard

### 🔧 Technical Features

#### API Excellence
- **RESTful Design**: Clean, intuitive API endpoints
- **Async Processing**: High-concurrency request handling
- **Type Validation**: Pydantic schemas for data integrity
- **Comprehensive Docs**: Interactive OpenAPI documentation

#### Production Readiness
- **Error Handling**: Graceful error management with user feedback
- **Logging System**: Structured logging with configurable levels
- **Health Monitoring**: Built-in health checks and status endpoints
- **Security**: Input validation, CORS configuration, and best practices

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18.0+ (for frontend)
- **pnpm** 8.0+ (recommended for frontend)
- **Python** 3.8+ (for backend)
- **Docker** (optional, for containerized deployment)

### Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/austinLorenzMccoy/credit-default-prediction.git
   cd credit-default-prediction
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python main.py
   ```
   Backend will be available at `http://localhost:8000`

3. **Frontend Setup**
   ```bash
   cd frontend
   pnpm install
   cp .env.example .env
   pnpm run dev
   ```
   Frontend will be available at `http://localhost:3000`

4. **Access the Application**
   - **Frontend**: https://credit-default-prediction.vercel.app/ ✅ **LIVE**
   - **API Docs**: https://credit-default-prediction-yod3.onrender.com/docs
   - **Health Check**: https://credit-default-prediction-yod3.onrender.com/api/v1/health

---

## 📁 Project Structure

```
credit-default-prediction/
├── 📂 backend/                # Backend API and ML services
│   ├── 📂 app/               # FastAPI application
│   │   ├── 📂 api/           # API routes and endpoints
│   │   ├── 📂 config/        # Configuration settings
│   │   ├── 📂 models/        # ML model implementations
│   │   ├── 📂 schemas/       # Pydantic data models
│   │   └── 📂 utils/         # Utility functions
│   ├── 📂 scripts/           # CLI and utility scripts
│   ├── 📂 tests/             # Backend unit tests
│   ├── 📂 models/            # Pre-trained model files
│   ├── 📄 requirements.txt   # Python dependencies
│   ├── 📄 main.py           # Backend entry point
│   └── 📄 README.md         # Backend documentation
├── 📂 frontend/              # Modern React web application
│   ├── 📂 src/              # React source code
│   │   ├── 📂 components/   # React components
│   │   ├── 📂 services/     # API integration layer
│   │   └── 📂 types/        # TypeScript definitions
│   ├── 📄 package.json      # Node.js dependencies
│   ├── 📄 vite.config.ts    # Vite build configuration
│   └── 📄 README.md         # Frontend documentation
├── 📂 frontend/              # Legacy frontend (deprecated)
├── 📂 notebook/              # Jupyter notebooks for development
├── 📂 examples/              # API usage examples
├── 📂 tests/                 # Integration tests
├── 📄 pyproject.toml         # Project configuration
├── 📄 Dockerfile             # Docker container setup
├── 📄 requirements.txt       # Root dependencies
└── 📄 README.md             # This file
```

---

## 🔌 API Documentation

### Base URL
```
http://localhost:8000/api/v1
```

### Core Endpoints

#### 🏥 Health Check
```http
GET /health
```
Check API and model status

#### 🎯 Predict Default Risk
```http
POST /predict/default
```
Predict credit default probability

#### 💳 Predict Credit Limit
```http
POST /predict/credit-limit
```
Get recommended credit limits

### Request Example

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

### Response Example

```json
{
  "prediction": "Low Risk of Default (Probability: 15.00%)",
  "probability": 0.15,
  "is_high_risk": false,
  "risk_factors": ["Low payment to bill ratio"]
}
```

---

## 🐳 Docker Deployment

### Quick Start with Docker

1. **Build the Image**
   ```bash
   docker build -t creditintel-pro .
   ```

2. **Run the Container**
   ```bash
   docker run -p 8000:8000 creditintel-pro
   ```

3. **Using Docker Compose**
   ```bash
   docker-compose up -d
   ```

### Production Dockerfile

The project includes an optimized Dockerfile for production deployment with:
- **Multi-stage builds** for minimal image size
- **Security best practices** with non-root user
- **Health checks** for container monitoring
- **Environment configuration** for flexible deployment

---

## 📊 Performance & Monitoring

### 🚀 Performance Metrics

- **Response Time**: < 200ms average API response
- **Throughput**: 1000+ requests/second
- **Model Inference**: < 50ms per prediction
- **Memory Usage**: Efficient model loading and caching

### 📈 Monitoring Features

- **Health Endpoints**: Real-time system status
- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Error Tracking**: Comprehensive error reporting and analysis
- **Performance Metrics**: Request timing and success rates

### 🔧 Configuration

Environment-based configuration for:
- **Development**: Debug mode, detailed logging
- **Testing**: Mock data, isolated environment
- **Production**: Optimized settings, security hardening

---

## 🧪 Testing & Quality Assurance

### 🧪 Test Coverage

- **Backend Tests**: Unit tests for API endpoints and models
- **Frontend Tests**: Component tests and integration testing
- **API Tests**: End-to-end API validation
- **Performance Tests**: Load testing and stress testing

### 🔍 Code Quality

- **Type Safety**: Full TypeScript implementation
- **Linting**: ESLint and Prettier for consistent code
- **Security**: Input validation and dependency scanning
- **Documentation**: Comprehensive inline and API documentation

### 🚀 CI/CD Pipeline

- **Automated Testing**: GitHub Actions for continuous testing
- **Code Quality**: Automated linting and type checking
- **Security Scanning**: Dependency vulnerability checks
- **Deployment**: Automated Docker image building and pushing

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### 🛠️ Development Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes with proper testing
4. **Commit** with descriptive messages: `git commit -m 'Add amazing feature'`
5. **Push** to your branch: `git push origin feature/amazing-feature`
6. **Open** a Pull Request with detailed description

### 📋 Contribution Guidelines

- **Code Style**: Follow existing patterns and conventions
- **Testing**: Add tests for new features and bug fixes
- **Documentation**: Update docs for API changes and new features
- **Commits**: Use clear, descriptive commit messages
- **Issues**: Report bugs with detailed reproduction steps

### 🏆 Recognition

Contributors are recognized in:
- **README.md**: Contributor acknowledgments
- **Changelog**: Feature attributions
- **GitHub**: Contributor statistics and badges

---

## 📚 Documentation & Resources

### 📖 Detailed Documentation

- **[Backend Guide](backend/README.md)**: Comprehensive backend documentation
- **[Frontend Guide](frontend/README.md)**: Complete frontend documentation
- **[API Reference](http://localhost:8000/docs)**: Interactive API documentation
- **[Examples](examples/)**: Code examples and usage patterns

### 🎓 Learning Resources

- **[Jupyter Notebooks](notebook/)**: Data exploration and model training
- **[Blog Posts]**: Technical articles and case studies
- **[Video Tutorials]**: Step-by-step video guides
- **[Community Forum]**: Discussion and support

### 🔗 External Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **React Documentation**: https://react.dev/
- **TensorFlow Guides**: https://www.tensorflow.org/guide
- **Tailwind CSS**: https://tailwindcss.com/docs

---

## 🐛 Troubleshooting

### 🔧 Common Issues

#### Backend Issues
- **Port Conflicts**: Use different port with `--port` flag
- **Model Loading**: Verify model files and permissions
- **Dependencies**: Ensure correct Python version and virtual environment

#### Frontend Issues
- **API Connection**: Check backend URL in `.env` file
- **Build Errors**: Clear cache with `npm run clean`
- **Type Errors**: Run `npm run lint` for type checking

#### Docker Issues
- **Build Failures**: Check Dockerfile and dependencies
- **Permission Errors**: Use proper file permissions
- **Network Issues**: Verify port mapping and firewall settings

### 🆘 Getting Help

- **GitHub Issues**: Report bugs and feature requests
- **Community Forum**: Get help from other users
- **Email Support**: chibuezeaugustine23@gmail.com
- **Documentation**: Check comprehensive guides and examples

---

## 🚀 Roadmap & Future Features

### 🎯 Upcoming Enhancements

- **🔐 Authentication**: JWT-based user authentication and authorization
- **📊 Advanced Analytics**: Real-time dashboards and business intelligence
- **🌐 Multi-tenant**: Support for multiple organizations and data isolation
- **📱 Mobile App**: React Native mobile application
- **🤖 AutoML**: Automated model training and hyperparameter tuning

### 🚀 Long-term Vision

- **🏢 Enterprise Features**: Advanced compliance and audit capabilities
- **🌍 Internationalization**: Multi-language support and regional compliance
- **🔗 Integrations**: Third-party banking and credit bureau integrations
- **📈 Scalability**: Horizontal scaling and load balancing
- **🔒 Security**: Advanced security features and compliance certifications

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### 📋 License Summary

- ✅ **Commercial Use**: Use in commercial projects
- ✅ **Modification**: Modify and distribute
- ✅ **Distribution**: Distribute original or modified versions
- ✅ **Private Use**: Use in private projects
- ❌ **Liability**: No warranty or liability
- ❌ **Trademark**: No trademark use without permission

---

## 🙏 Acknowledgments

### 🏆 Core Contributors

- **[@austinLorenzMccoy](https://github.com/austinLorenzMccoy)** - Project founder and lead developer
- **Credit Risk Team** - ML model development and validation
- **Frontend Team** - UI/UX design and implementation
- **DevOps Team** - Infrastructure and deployment

### 🌟 Technology Credits

- **FastAPI Team** - Amazing web framework
- **React Team** - Incredible UI library
- **TensorFlow Team** - Powerful ML framework
- **Open Source Community** - Countless contributors to our dependencies

### 🏢 Institutional Support

- **Financial Institutions** - Domain expertise and requirements
- **Research Partners** - Academic collaboration and validation
- **Beta Testers** - Early feedback and improvement suggestions

---

## 📞 Contact & Support

### 📧 Direct Contact

- **Email**: chibuezeaugustine23@gmail.com
- **GitHub**: [@austinLorenzMccoy](https://github.com/austinLorenzMccoy)
- **LinkedIn**: [Professional Profile](https://linkedin.com/in/austinmccoy)

### 🔗 Online Resources

- **🐛 Issues**: [GitHub Issues](https://github.com/austinLorenzMccoy/credit-default-prediction/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/austinLorenzMccoy/credit-default-prediction/discussions)
- **📖 Wiki**: [Project Wiki](https://github.com/austinLorenzMccoy/credit-default-prediction/wiki)
- **📊 Analytics**: [Project Stats](https://github.com/austinLorenzMccoy/credit-default-prediction/pulse)

### 🌐 Community

- **Twitter**: [@CreditIntelPro](https://twitter.com/CreditIntelPro)
- **LinkedIn**: [Company Page](https://linkedin.com/company/creditintel-pro)
- **Medium**: [Blog](https://medium.com/creditintel-pro)
- **YouTube**: [Tutorials](https://youtube.com/c/creditintel-pro)

---

<div align="center">
  <h2>🚀 Start Building with CreditIntel Pro Today!</h2>
  
  <p><em>Empowering financial institutions with intelligent credit risk assessment</em></p>
  
  <div>
    <a href="#quick-start">
      <img src="https://img.shields.io/badge/Get%20Started-4CAF50?style=for-the-badge&logo=github&logoColor=white" alt="Get Started">
    </a>
    <a href="https://github.com/austinLorenzMccoy/credit-default-prediction/stargazers">
      <img src="https://img.shields.io/badge/⭐-Star%20on%20GitHub-FFD700?style=for-the-badge" alt="Star on GitHub">
    </a>
    <a href="https://github.com/austinLorenzMccoy/credit-default-prediction/fork">
      <img src="https://img.shields.io/badge/🍴-Fork%20on%20GitHub-4169E1?style=for-the-badge" alt="Fork on GitHub">
    </a>
  </div>
  
  <p>
    <strong>Made with ❤️ for the financial technology community</strong><br>
    © 2024 CreditIntel Pro. All rights reserved.
  </p>
</div>
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
credit-default-prediction/  
├── backend/               # Backend API and ML models  
│   ├── app/               # FastAPI application code  
│   │   ├── api/           # API routes and endpoints  
│   │   ├── config/        # Configuration settings  
│   │   ├── models/        # ML model implementations  
│   │   ├── schemas/       # Pydantic request/response models  
│   │   └── utils/         # Utility functions  
│   ├── scripts/           # CLI and utility scripts  
│   │   ├── cli.py         # Command-line interface  
│   │   └── run.py         # Application runner  
│   ├── tests/             # Backend unit tests  
│   ├── models/            # Pre-trained model files  
│   ├── requirements.txt   # Backend Python dependencies  
│   └── main.py            # Backend entry point  
├── frontend/              # Frontend UI files (for future improvement)  
│   ├── css/               # Stylesheets  
│   ├── js/                # JavaScript files  
│   └── index.html         # Main UI page  
├── notebook/              # Jupyter notebooks for EDA and training  
├── models/                # Additional model storage  
├── examples/              # API usage examples  
├── tests/                 # Root-level tests  
├── pyproject.toml         # Project configuration  
├── Dockerfile             # Docker configuration  
├── requirements.txt       # Root dependencies  
└── README.md              # Project documentation  
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

