import React from 'react';
import { Download, CheckCircle2, AlertTriangle, TrendingUp, ChevronRight, BarChart3, Lightbulb } from 'lucide-react';
import { motion } from 'motion/react';

export function CreditLimitDetail() {
  return (
    <div className="space-y-stack-lg">
      {/* Page Header */}
      <div className="flex justify-between items-end border-b-2 border-slate-blue pb-stack-sm">
        <div>
          <nav className="flex items-center gap-2 mb-4 font-mono text-[10px] text-slate-500 uppercase tracking-widest">
            <span>Profiling</span>
            <ChevronRight size={12} />
            <span className="text-copper">Credit Limit Detail</span>
          </nav>
          <h2 className="text-4xl font-heading text-ink">Limit Recommendation</h2>
          <p className="text-body text-slate-500 mt-1">Detailed analysis for Global Logistics Corp (GLC-8829)</p>
        </div>
        <div className="flex gap-4">
          <button className="px-6 h-[44px] border-[1.5px] border-copper text-copper font-mono text-xs font-bold uppercase tracking-widest rounded-lg hover:bg-mist transition-colors">
            View API Docs
          </button>
          <button className="px-6 h-[44px] bg-copper text-white font-mono text-xs font-bold uppercase tracking-widest rounded-lg hover:opacity-90 shadow-ambient transition-all active:scale-95">
            Approve Increase
          </button>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-gutter">
        {/* Comparison Card */}
        <div className="col-span-12 lg:col-span-8 bg-white border border-rule border-t-4 border-t-slate-blue rounded-lg p-8 shadow-ambient">
          <div className="flex justify-between items-center mb-12">
            <h3 className="font-mono text-xs font-bold text-ink uppercase tracking-widest">Limit Comparison</h3>
            <span className="bg-copper/10 text-copper px-3 py-1 rounded-full text-[11px] font-bold font-mono uppercase tracking-tight">
              Adjustment: x1.5
            </span>
          </div>
          
          <div className="flex items-end gap-16 h-64 pb-8">
            {/* Current Limit Bar */}
            <div className="flex-1 flex flex-col items-center gap-4 group">
              <motion.div 
                initial={{ height: 0 }}
                animate={{ height: '66.6%' }}
                className="w-24 bg-slate-200 hover:bg-slate-300 transition-all rounded-t-lg relative"
              >
                <span className="absolute -top-10 left-1/2 -translate-x-1/2 font-mono text-slate-500 text-lg">$100k</span>
              </motion.div>
              <span className="font-mono text-[11px] text-slate-400 uppercase tracking-widest">Current</span>
            </div>
            
            {/* Recommended Limit Bar */}
            <div className="flex-1 flex flex-col items-center gap-4 group">
              <motion.div 
                initial={{ height: 0 }}
                animate={{ height: '100%' }}
                className="w-24 bg-copper hover:opacity-90 transition-all rounded-t-lg relative"
              >
                <span className="absolute -top-10 left-1/2 -translate-x-1/2 font-mono text-copper text-2xl font-black">$150k</span>
                <div className="absolute inset-0 bg-gradient-to-t from-black/10 to-transparent opacity-20"></div>
              </motion.div>
              <span className="font-mono text-[11px] text-ink font-bold uppercase tracking-widest">Recommended</span>
            </div>
          </div>

          <div className="mt-8 pt-8 border-t border-rule grid grid-cols-3 gap-8">
            {[
              { label: 'Delta Value', value: '+$50,000', color: 'text-ink' },
              { label: 'Risk Exposure', value: 'Low (0.12%)', color: 'text-sage' },
              { label: 'Confidence', value: '94.8%', color: 'text-amber' },
            ].map((stat, idx) => (
              <div key={idx}>
                <p className="text-[10px] font-mono text-slate-500 uppercase tracking-widest mb-1">{stat.label}</p>
                <p className={`text-2xl font-mono font-bold ${stat.color}`}>{stat.value}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Factors Card */}
        <div className="col-span-12 lg:col-span-4 bg-white border border-rule border-t-4 border-t-sage rounded-lg p-8 shadow-ambient">
          <h3 className="font-mono text-xs font-bold text-ink uppercase tracking-widest mb-8">Recommendation Factors</h3>
          <ul className="space-y-6">
            {[
              { title: 'Cash Flow Consistency', desc: '24 months of uninterrupted positive quarterly net growth.', icon: CheckCircle2, color: 'text-sage', bg: 'bg-sage/10' },
              { title: 'Market Sentiment Index', desc: 'Top-tier performance in Logistics Sector benchmark.', icon: CheckCircle2, color: 'text-sage', bg: 'bg-sage/10' },
              { title: 'Account Longevity', desc: 'Customer tenure exceeds 5-year primary risk threshold.', icon: CheckCircle2, color: 'text-sage', bg: 'bg-sage/10' },
              { title: 'Macro Environment', desc: 'Regional inflation rates may impact secondary margins.', icon: AlertTriangle, color: 'text-amber', bg: 'bg-amber/10' },
            ].map((factor, idx) => (
              <li key={idx} className="flex items-start gap-4">
                <div className={`mt-1 w-6 h-6 rounded-full ${factor.bg} ${factor.color} flex items-center justify-center`}>
                  <factor.icon size={14} className="font-bold" />
                </div>
                <div>
                  <p className="font-heading font-bold text-ink text-sm leading-tight">{factor.title}</p>
                  <p className="text-xs text-slate-500 mt-1 leading-relaxed">{factor.desc}</p>
                </div>
              </li>
            ))}
          </ul>
        </div>

        {/* Factor Breakdown */}
        <div className="col-span-12 bg-white border border-rule rounded-lg p-8 shadow-ambient">
          <h3 className="font-mono text-xs font-bold text-ink uppercase tracking-widest mb-10">Adjustment Factor Breakdown (Weight %)</h3>
          <div className="space-y-10">
            {[
              { label: 'Revenue Growth Velocity', weight: 40 },
              { label: 'Historical Payment Reliability', weight: 25 },
              { label: 'Industry Benchmarking (Logistics)', weight: 20 },
              { label: 'External Bureau Integration', weight: 15 },
            ].map((item, idx) => (
              <div key={idx}>
                <div className="flex justify-between items-end mb-3">
                  <span className="font-heading font-bold text-ink">{item.label}</span>
                  <span className="font-mono text-copper font-bold text-xl">{item.weight}%</span>
                </div>
                <div className="h-3 w-full bg-mist rounded-full overflow-hidden">
                  <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: `${item.weight}%` }}
                    className="h-full bg-copper rounded-full"
                  ></motion.div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom Insight Row */}
        <div className="col-span-12 lg:col-span-6 bg-slate-blue text-white rounded-lg p-8 flex gap-8 items-center shadow-lg">
          <div className="w-24 h-24 flex-shrink-0 bg-white/10 rounded-full flex items-center justify-center backdrop-blur-sm">
            <Lightbulb className="text-white" size={40} />
          </div>
          <div>
            <h4 className="font-heading text-2xl mb-2">Insight: Upside Potential</h4>
            <p className="text-slate-300 text-sm leading-relaxed mb-4">GLC's current utilization is at 88%. By increasing the limit to $150k, we project a 12% increase in transaction volume within 90 days with negligible risk impact.</p>
            <a href="#" className="font-mono text-[10px] uppercase font-bold tracking-widest underline decoration-white/30 hover:text-copper transition-colors">Read Detailed Forecast</a>
          </div>
        </div>

        <div className="col-span-12 lg:col-span-6 border border-rule bg-white rounded-lg p-8 flex items-center justify-between shadow-ambient">
          <div className="space-y-2">
            <h4 className="font-mono text-[10px] text-slate-500 uppercase tracking-widest">Predicted Loss Rate</h4>
            <p className="text-4xl font-heading text-ink">0.08%</p>
            <span className="bg-sage/10 text-sage px-3 py-1 rounded-full text-[10px] font-bold font-mono uppercase tracking-widest border border-sage/20">Elite Grade</span>
          </div>
          <div className="relative w-32 h-16">
            <svg className="w-full h-full" viewBox="0 0 100 50">
              <path 
                d="M 10,50 A 40,40 0 0,1 90,50" 
                fill="none" 
                stroke="#F0E6D3" 
                strokeWidth="10"
              />
              <motion.path 
                initial={{ strokeDashoffset: 125 }}
                animate={{ strokeDashoffset: 35 }}
                d="M 10,50 A 40,40 0 0,1 90,50" 
                fill="none" 
                stroke="#4A7C59" 
                strokeDasharray="125" 
                strokeWidth="10"
              />
            </svg>
            <div className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[10px] font-mono text-slate-300 uppercase tracking-tighter">Low Risk</div>
          </div>
        </div>
      </div>
    </div>
  );
}
