export type Page = 'landing' | 'overview' | 'profiling' | 'history' | 'api-status' | 'default-detail' | 'limit-detail';

export interface Prediction {
  id: string;
  date: string;
  entityName: string;
  type: 'DEFAULT RISK' | 'CREDIT LIMIT';
  riskScore: number;
  limit?: number;
  status: 'APPROVED' | 'HIGH RISK' | 'RE-EVALUATE' | 'MODERATE' | 'SETTLED' | 'PARTIAL';
  variance?: number;
}

export interface CustomerProfile {
  age: number;
  gender: 'Male' | 'Female' | 'Other';
  education: 'Graduate School' | 'University' | 'High School';
  maritalStatus: 'Married' | 'Single' | 'Divorced';
  desiredLimit: number;
  payStatus: 'Paid on Time' | '1 Month Delay' | '2 Month Delay';
  billAmount: number;
  paymentAmount: number;
}
