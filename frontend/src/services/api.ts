/**
 * API Service for Credit Default Prediction Backend
 * Handles all communication with the FastAPI backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8003';

export interface ApiPredictionRequest {
  credit_limit: number;
  age: number;
  gender: number; // 1=male, 2=female
  education: number; // 1=graduate, 2=university, 3=high school, 4=others
  marital_status: number; // 1=married, 2=single, 3=others
  payment_status: number; // 0=paid duly, 1=1 month delay, etc.
  bill_amount: number;
  payment_amount: number;
}

export interface DefaultPredictionResponse {
  prediction: string;
  probability: number;
  is_high_risk: boolean;
  risk_factors: string[];
}

export interface CreditLimitPredictionResponse {
  predicted_credit_limit: number;
  adjustment_factor: number;
  recommendation_factors: string[];
}

export interface HealthResponse {
  status: string;
  model_status: {
    predictor_ready: boolean;
  };
}

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Health check endpoint
  async healthCheck(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/api/v1/health');
  }

  // Predict credit default risk
  async predictDefaultRisk(data: ApiPredictionRequest): Promise<DefaultPredictionResponse> {
    return this.request<DefaultPredictionResponse>('/api/v1/predict/default', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Predict credit limit
  async predictCreditLimit(data: ApiPredictionRequest): Promise<CreditLimitPredictionResponse> {
    return this.request<CreditLimitPredictionResponse>('/api/v1/predict/credit-limit', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Convert frontend CustomerProfile to API request format
  private convertCustomerProfileToApiRequest(profile: any): ApiPredictionRequest {
    const genderMap: { [key: string]: number } = {
      'Male': 1,
      'Female': 2,
      'Other': 2,
    };

    const educationMap: { [key: string]: number } = {
      'Graduate School': 1,
      'University': 2,
      'High School': 3,
    };

    const maritalStatusMap: { [key: string]: number } = {
      'Married': 1,
      'Single': 2,
      'Divorced': 3,
      'Other': 3,
    };

    const payStatusMap: { [key: string]: number } = {
      'Paid on Time': 0,
      '1 Month Delay': 1,
      '2 Month Delay': 2,
    };

    return {
      credit_limit: profile.desiredLimit || 100000,
      age: profile.age,
      gender: genderMap[profile.gender] || 2,
      education: educationMap[profile.education] || 2,
      marital_status: maritalStatusMap[profile.maritalStatus] || 2,
      payment_status: payStatusMap[profile.payStatus] || 0,
      bill_amount: profile.billAmount,
      payment_amount: profile.paymentAmount,
    };
  }

  // Predict default risk from customer profile
  async predictDefaultRiskFromProfile(profile: any): Promise<DefaultPredictionResponse> {
    const apiRequest = this.convertCustomerProfileToApiRequest(profile);
    return this.predictDefaultRisk(apiRequest);
  }

  // Predict credit limit from customer profile
  async predictCreditLimitFromProfile(profile: any): Promise<CreditLimitPredictionResponse> {
    const apiRequest = this.convertCustomerProfileToApiRequest(profile);
    return this.predictCreditLimit(apiRequest);
  }
}

// Create singleton instance
export const apiService = new ApiService();

// Export types for use in components
export type { ApiService };
