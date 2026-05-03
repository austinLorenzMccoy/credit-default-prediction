<div align="center">
  <h1>🏦 CreditIntel Pro Frontend</h1>
  <p><em>Sophisticated React TypeScript frontend for credit risk assessment and prediction</em></p>
  <img src="https://img.shields.io/badge/React-19.0.1-blue" alt="React">
  <img src="https://img.shields.io/badge/TypeScript-5.8-blue" alt="TypeScript">
  <img src="https://img.shields.io/badge/Vite-6.2.3-orange" alt="Vite">
  <img src="https://img.shields.io/badge/Tailwind-4.1.14-38B2AC" alt="Tailwind CSS">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</div>

## 🎯 Overview

CreditIntel Pro Frontend is a modern, responsive web application that provides an intuitive interface for credit risk assessment and prediction. Built with React, TypeScript, and Tailwind CSS, it offers a seamless user experience for financial institutions to evaluate customer credit profiles and make data-driven decisions.

### ✨ Key Features

- **🎨 Modern UI/UX**: Beautiful, professional interface with smooth animations and transitions
- **📱 Responsive Design**: Fully responsive layout that works on all devices
- **⚡ Real-time API Integration**: Seamless connection to backend ML services
- **🔍 Customer Profiling**: Comprehensive forms for capturing customer financial data
- **📊 Risk Visualization**: Interactive charts and visual representations of risk scores
- **📋 Prediction History**: Track and analyze historical predictions and trends
- **🔔 Error Handling**: Robust error handling with user-friendly notifications
- **🎯 Type Safety**: Full TypeScript implementation for better code reliability

---

## 🛠️ Tech Stack

| Technology | Version | Description |
|------------|---------|-------------|
| **React** | 19.0.1 | UI library for building interactive interfaces |
| **TypeScript** | 5.8.2 | Static type checking for better code quality |
| **Vite** | 6.2.3 | Fast build tool and development server |
| **Tailwind CSS** | 4.1.14 | Utility-first CSS framework for styling |
| **Lucide React** | 0.546.0 | Beautiful icon library |
| **Motion** | 12.23.24 | Animation library for smooth transitions |
| **Express** | 4.21.2 | Backend server for development |

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** 18.0 or higher
- **npm** 9.0 or higher
- **Backend API** running on port 8003 (default)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/austinLorenzMccoy/credit-default-prediction.git
   cd credit-default-prediction/frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Environment Configuration:**
   ```bash
   cp .env.example .env
   # Edit .env file with your API configuration
   ```

4. **Start the development server:**
   ```bash
   npm run dev
   ```

5. **Open your browser:**
   Navigate to `http://localhost:3000`

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# API Configuration
VITE_API_URL=http://localhost:8003

# Development settings
VITE_NODE_ENV=development
```

### Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start development server on port 3000 |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run clean` | Clean build artifacts |
| `npm run lint` | Run TypeScript type checking |

---

## 📁 Project Structure

```
frontend/
├── public/                 # Static assets
├── src/
│   ├── components/          # React components
│   │   ├── Dashboard.tsx    # Main customer profiling interface
│   │   ├── LandingPage.tsx  # Landing page component
│   │   ├── Overview.tsx     # Executive overview
│   │   ├── DefaultRiskDetail.tsx  # Risk prediction details
│   │   ├── CreditLimitDetail.tsx  # Credit limit recommendations
│   │   ├── PredictionHistory.tsx  # Historical predictions
│   │   └── layout.tsx       # Main layout component
│   ├── services/
│   │   └── api.ts           # API service layer
│   ├── types.ts             # TypeScript type definitions
│   ├── App.tsx              # Main application component
│   ├── main.tsx             # Application entry point
│   └── index.css            # Global styles
├── package.json             # Dependencies and scripts
├── tsconfig.json            # TypeScript configuration
├── vite.config.ts           # Vite build configuration
└── README.md                # This file
```

---

## 🔌 API Integration

The frontend integrates with the backend API through a centralized service layer:

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Check API health status |
| `/api/v1/predict/default` | POST | Predict credit default risk |
| `/api/v1/predict/credit-limit` | POST | Recommend credit limits |

### Usage Example

```typescript
import { apiService } from './services/api';

// Predict default risk
const prediction = await apiService.predictDefaultRiskFromProfile(customerProfile);

// Predict credit limit
const limit = await apiService.predictCreditLimitFromProfile(customerProfile);
```

---

## 🎨 UI Components

### Core Features

1. **Customer Intelligence Dashboard**
   - Comprehensive customer profiling forms
   - Real-time risk assessment
   - Interactive data visualization

2. **Risk Prediction Interface**
   - Default probability calculation
   - Risk factor analysis
   - Historical trend tracking

3. **Credit Limit Recommendations**
   - Dynamic limit suggestions
   - Adjustment factor explanations
   - Approval workflow

4. **Executive Overview**
   - Portfolio-level insights
   - Performance metrics
   - System health monitoring

---

## 🔧 Development

### Code Style

- **TypeScript**: Strict type checking enabled
- **ESLint**: Code quality and consistency
- **Prettier**: Code formatting (configured)
- **Tailwind CSS**: Utility-first styling approach

### Component Architecture

- **Functional Components**: Using React hooks and modern patterns
- **Type Safety**: Full TypeScript implementation
- **Modular Design**: Reusable, composable components
- **State Management**: Local state with hooks, API integration through services

### Performance Optimizations

- **Code Splitting**: Automatic with Vite
- **Lazy Loading**: Components loaded on demand
- **Tree Shaking**: Unused code elimination
- **Asset Optimization**: Image and resource optimization

---

## 🚀 Deployment

### Production Build

```bash
npm run build
```

### Preview Build

```bash
npm run preview
```

### Environment Setup

For production deployment, ensure:

1. **API URL**: Set `VITE_API_URL` to production backend endpoint
2. **HTTPS**: Use HTTPS for secure API communication
3. **CORS**: Backend configured to allow frontend origin
4. **Build Optimization**: Production build optimized for performance

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Guidelines

- Follow TypeScript best practices
- Use descriptive component and variable names
- Add proper error handling and loading states
- Ensure responsive design for all components
- Write meaningful commit messages

---

## 🐛 Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Verify backend is running on correct port
   - Check `VITE_API_URL` in `.env` file
   - Ensure CORS is configured on backend

2. **Build Errors**
   - Clear node_modules: `rm -rf node_modules package-lock.json`
   - Reinstall dependencies: `npm install`
   - Check TypeScript types: `npm run lint`

3. **Development Server Issues**
   - Check port availability (default: 3000)
   - Verify Node.js version compatibility
   - Clear browser cache and restart

---

## 📄 License

This project is licensed under the MIT License - see the main project LICENSE file for details.

---

## 🙏 Acknowledgments

- **React Team** - For the amazing React framework
- **Tailwind CSS** - For the utility-first CSS framework
- **Vite** - For the blazing fast build tool
- **Lucide** - For the beautiful icon library

---

## 📞 Support

For support and questions:

- 📧 Email: chibuezeaugustine23@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/austinLorenzMccoy/credit-default-prediction/issues)
- 📖 Documentation: [Project Wiki](https://github.com/austinLorenzMccoy/credit-default-prediction/wiki)

---

<div align="center">
  <p>Made with ❤️ for financial institutions</p>
  <p>© 2024 CreditIntel Pro. All rights reserved.</p>
</div>
