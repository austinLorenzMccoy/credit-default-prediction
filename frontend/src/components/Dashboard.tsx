import React, { useState } from 'react';
import { Search, Activity, ShieldCheck, TrendingUp, Info, ChevronRight, UserCircle2, Loader2 } from 'lucide-react';
import { CustomerProfile, Page } from '../types';
import { motion } from 'motion/react';
import { apiService, DefaultPredictionResponse, CreditLimitPredictionResponse } from '../services/api';

export function Dashboard({ onPredict }: { onPredict: (p: Page) => void }) {
  const [profile, setProfile] = useState<CustomerProfile>({
    age: 34,
    gender: 'Female',
    education: 'Graduate School',
    maritalStatus: 'Married',
    desiredLimit: 150000,
    payStatus: 'Paid on Time',
    billAmount: 45200,
    paymentAmount: 10000
  });
  
  const [isLoading, setIsLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [defaultPrediction, setDefaultPrediction] = useState<DefaultPredictionResponse | null>(null);
  const [creditLimitPrediction, setCreditLimitPrediction] = useState<CreditLimitPredictionResponse | null>(null);

  const handlePredictDefaultRisk = async () => {
    setIsLoading(true);
    setApiError(null);
    
    try {
      const prediction = await apiService.predictDefaultRiskFromProfile(profile);
      setDefaultPrediction(prediction);
      onPredict('default-detail');
    } catch (error) {
      setApiError('Failed to predict default risk. Please try again.');
      console.error('Prediction error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePredictCreditLimit = async () => {
    setIsLoading(true);
    setApiError(null);
    
    try {
      const prediction = await apiService.predictCreditLimitFromProfile(profile);
      setCreditLimitPrediction(prediction);
      onPredict('limit-detail');
    } catch (error) {
      setApiError('Failed to predict credit limit. Please try again.');
      console.error('Prediction error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <header className="mb-10">
        <h2 className="font-display text-4xl text-ink mb-2">Customer Intelligence</h2>
        <p className="text-body text-slate-blue/70">Execute precision risk modeling for high-value credit applications.</p>
      </header>

      <div className="grid grid-cols-12 gap-8">
        {/* INPUT FORM */}
        <section className="col-span-12 lg:col-span-8">
          <div className="bg-white rounded-xl border border-rule p-8 shadow-ambient border-t-4 border-t-slate-blue">
            <div className="flex items-center justify-between mb-8 border-b border-rule pb-4">
              <h3 className="font-mono text-lg font-bold text-ink flex items-center gap-2 uppercase tracking-tight">
                <Activity className="text-copper" size={20} />
                INPUT FORM — Customer Financial Profile
              </h3>
              <span className="font-mono text-[10px] text-slate-400">SESSION ID: CI-99284</span>
            </div>
            
            <form className="space-y-8" onSubmit={(e) => { e.preventDefault(); handlePredictDefaultRisk(); }}>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Demographics */}
                <div className="space-y-4">
                  <div>
                    <label className="block font-mono text-[13px] text-slate-blue mb-1.5 uppercase tracking-wide">Age</label>
                    <input 
                      type="number" 
                      className="w-full h-11 border border-rule rounded-lg px-4 font-body focus:ring-2 focus:ring-copper focus:ring-offset-1 outline-none transition-all"
                      value={profile.age}
                      onChange={e => setProfile({...profile, age: parseInt(e.target.value)})}
                    />
                  </div>
                  <div>
                    <label className="block font-mono text-[13px] text-slate-blue mb-1.5 uppercase tracking-wide">Gender</label>
                    <select 
                      className="w-full h-11 border border-rule rounded-lg px-4 font-body focus:ring-2 focus:ring-copper focus:ring-offset-1 outline-none appearance-none bg-white cursor-pointer"
                      value={profile.gender}
                      onChange={e => setProfile({...profile, gender: e.target.value as any})}
                    >
                      <option>Female</option>
                      <option>Male</option>
                      <option>Other</option>
                    </select>
                  </div>
                  <div>
                    <label className="block font-mono text-[13px] text-slate-blue mb-1.5 uppercase tracking-wide">Education</label>
                    <select 
                      className="w-full h-11 border border-rule rounded-lg px-4 font-body focus:ring-2 focus:ring-copper focus:ring-offset-1 outline-none appearance-none bg-white cursor-pointer"
                      value={profile.education}
                      onChange={e => setProfile({...profile, education: e.target.value as any})}
                    >
                      <option>Graduate School</option>
                      <option>University</option>
                      <option>High School</option>
                    </select>
                  </div>
                  <div>
                    <label className="block font-mono text-[13px] text-slate-blue mb-1.5 uppercase tracking-wide">Marital Status</label>
                    <select 
                      className="w-full h-11 border border-rule rounded-lg px-4 font-body focus:ring-2 focus:ring-copper focus:ring-offset-1 outline-none appearance-none bg-white cursor-pointer"
                      value={profile.maritalStatus}
                      onChange={e => setProfile({...profile, maritalStatus: e.target.value as any})}
                    >
                      <option>Married</option>
                      <option>Single</option>
                      <option>Divorced</option>
                    </select>
                  </div>
                </div>

                {/* Financials */}
                <div className="space-y-4">
                  <div>
                    <label className="block font-mono text-[13px] text-slate-blue mb-1.5 uppercase tracking-wide">Desired Credit Limit</label>
                    <div className="relative">
                      <span className="absolute left-4 top-1/2 -translate-y-1/2 font-mono text-slate-400">₦</span>
                      <input 
                        type="text" 
                        className="w-full h-11 border border-rule rounded-lg pl-8 pr-4 font-mono text-sm focus:ring-2 focus:ring-copper focus:ring-offset-1 outline-none"
                        value={profile.desiredLimit.toLocaleString()}
                        onChange={e => setProfile({...profile, desiredLimit: parseInt(e.target.value.replace(/,/g, '')) || 0})}
                      />
                    </div>
                  </div>
                  <div>
                    <label className="block font-mono text-[13px] text-slate-blue mb-1.5 uppercase tracking-wide">Pay Status (Last Month)</label>
                    <select 
                      className="w-full h-11 border border-rule rounded-lg px-4 font-body focus:ring-2 focus:ring-copper focus:ring-offset-1 outline-none appearance-none bg-white cursor-pointer"
                      value={profile.payStatus}
                      onChange={e => setProfile({...profile, payStatus: e.target.value as any})}
                    >
                      <option>Paid on Time</option>
                      <option>1 Month Delay</option>
                      <option>2 Month Delay</option>
                    </select>
                  </div>
                  <div>
                    <label className="block font-mono text-[13px] text-slate-blue mb-1.5 uppercase tracking-wide">Bill Amount</label>
                    <div className="relative">
                      <span className="absolute left-4 top-1/2 -translate-y-1/2 font-mono text-slate-400">₦</span>
                      <input 
                        type="text" 
                        className="w-full h-11 border border-rule rounded-lg pl-8 pr-4 font-mono text-sm focus:ring-2 focus:ring-copper focus:ring-offset-1 outline-none"
                        value={profile.billAmount.toLocaleString()}
                        onChange={e => setProfile({...profile, billAmount: parseInt(e.target.value.replace(/,/g, '')) || 0})}
                      />
                    </div>
                  </div>
                  <div>
                    <label className="block font-mono text-[13px] text-slate-blue mb-1.5 uppercase tracking-wide">Payment Amount</label>
                    <div className="relative">
                      <span className="absolute left-4 top-1/2 -translate-y-1/2 font-mono text-slate-400">₦</span>
                      <input 
                        type="text" 
                        className="w-full h-11 border border-rule rounded-lg pl-8 pr-4 font-mono text-sm focus:ring-2 focus:ring-copper focus:ring-offset-1 outline-none"
                        value={profile.paymentAmount.toLocaleString()}
                        onChange={e => setProfile({...profile, paymentAmount: parseInt(e.target.value.replace(/,/g, '')) || 0})}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {apiError && (
                <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-red-700 text-sm font-mono">{apiError}</p>
                </div>
              )}
              
              <div className="pt-6 border-t border-rule flex justify-end gap-4">
                <button 
                  type="reset"
                  className="px-6 py-2.5 border-[1.5px] border-copper text-copper font-mono uppercase tracking-widest text-xs rounded-lg hover:bg-mist transition-colors"
                  disabled={isLoading}
                >
                  Clear Profile
                </button>
                <button 
                  type="submit"
                  className="px-8 py-2.5 bg-copper text-white font-mono uppercase tracking-widest text-xs rounded-lg hover:opacity-90 shadow-sm transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="animate-spin" size={16} />
                      Processing...
                    </>
                  ) : (
                    'Calculate Risk Prediction'
                  )}
                </button>
              </div>
            </form>
          </div>
        </section>

        {/* Quick Results Aside */}
        <aside className="col-span-12 lg:col-span-4 space-y-6">
          {/* Result Card 1: Default Risk */}
          <div className="bg-white rounded-xl border border-rule p-6 shadow-ambient overflow-hidden relative group cursor-pointer" onClick={() => onPredict('default-detail')}>
            <div className="flex items-center justify-between mb-6">
              <span className="font-mono text-[11px] uppercase tracking-wider text-slate-blue font-bold">Default Risk</span>
              <span className="px-2.5 py-1 bg-sage/10 text-sage text-[10px] font-bold font-mono rounded-full uppercase">Low Risk</span>
            </div>
            <div className="flex flex-col items-center mb-6">
              <div className="relative w-40 h-20 mb-4 overflow-hidden">
                <div className="w-40 h-40 border-[16px] border-mist rounded-full absolute top-0"></div>
                {/* Simulated partial gauge */}
                <div 
                  className="w-40 h-40 border-[16px] border-sage rounded-full absolute top-0 border-b-transparent border-l-transparent border-r-transparent transform rotate-[45deg]"
                ></div>
                <div className="absolute bottom-0 w-full text-center">
                  <span className="font-mono text-4xl text-sage font-bold tracking-tight">15%</span>
                </div>
              </div>
              <p className="text-body text-sm text-slate-500 text-center px-4">Probability of default based on behavioral payment trends.</p>
            </div>
            <div className="bg-mist p-4 rounded-lg flex items-center gap-3">
              <ShieldCheck className="text-sage shrink-0" size={20} />
              <span className="font-mono text-xs font-bold text-ink">Strong repayment profile detected.</span>
            </div>
          </div>

          {/* Result Card 2: Credit Limit */}
          <div className="bg-white rounded-xl border border-rule p-6 shadow-ambient border-t-4 border-t-copper group cursor-pointer" onClick={() => onPredict('limit-detail')}>
            <span className="font-mono text-[11px] uppercase tracking-wider text-slate-blue font-bold block mb-4">Recommended Limit</span>
            <div className="flex items-baseline gap-2 mb-2">
              <span className="font-mono text-3xl text-ink font-bold tracking-tight">₦150,000</span>
              <span className="font-mono text-sage text-sm font-bold flex items-center gap-0.5">
                <TrendingUp size={14} />
                +50%
              </span>
            </div>
            <p className="text-body text-xs text-slate-400 mb-6 italic">Adjustment from previous limit: ₦100,000</p>
            <div className="space-y-3">
              <div className="flex justify-between items-center text-xs">
                <span className="font-mono text-slate-500 uppercase tracking-tight">Liquidity Score</span>
                <span className="font-mono text-ink font-bold">88/100</span>
              </div>
              <div className="w-full bg-mist h-1.5 rounded-full overflow-hidden">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: '88%' }}
                  className="bg-copper h-full"
                ></motion.div>
              </div>
            </div>
          </div>

          {/* Institutional Insight */}
          <div className="p-6 rounded-xl bg-slate-blue text-white shadow-lg relative overflow-hidden group">
            <div className="relative z-10">
              <h4 className="font-heading text-xl mb-2">Institutional Insight</h4>
              <p className="text-body text-sm opacity-80 leading-relaxed mb-4">
                This profile shows exceptional stability in "Education" and "Pay Status" sectors. We recommend immediate approval for the requested increase.
              </p>
              <button 
                onClick={() => onPredict('default-detail')}
                className="font-mono text-[10px] uppercase tracking-widest flex items-center gap-2 group-hover:translate-x-1 transition-transform"
              >
                View Detailed Report <ChevronRight size={14} />
              </button>
            </div>
            <UserCircle2 className="absolute -right-10 -bottom-10 opacity-10 text-white" size={160} />
          </div>
        </aside>
      </div>

      {/* Footer Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-12">
        {[
          { label: 'Model Version', value: 'XGB-CREDIT v4.2' },
          { label: 'Inference Time', value: '142ms' },
          { label: 'Last Profile Scan', value: 'Just Now' },
        ].map((stat, idx) => (
          <div key={idx} className="bg-white/50 backdrop-blur-sm border border-rule p-6 rounded-xl">
            <span className="font-mono text-[10px] text-slate-500 uppercase tracking-widest">{stat.label}</span>
            <p className="font-mono text-lg text-ink font-bold mt-1">{stat.value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
