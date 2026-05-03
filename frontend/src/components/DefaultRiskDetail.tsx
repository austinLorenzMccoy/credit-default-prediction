import React from 'react';
import { Download, History, Info, AlertTriangle, CheckCircle2, TrendingDown, Verified, ChevronRight, Activity, TrendingUp } from 'lucide-react';
import { motion } from 'motion/react';

export function DefaultRiskDetail() {
  return (
    <div className="space-y-stack-lg">
      {/* Page Header */}
      <div className="flex justify-between items-end border-b-2 border-slate-blue pb-stack-sm">
        <div>
          <nav className="flex items-center space-x-2 font-mono text-[10px] text-slate-500 mb-2 uppercase tracking-widest">
            <span>Dashboard</span>
            <ChevronRight size={12} />
            <span className="text-copper">Prediction Analysis</span>
          </nav>
          <h2 className="font-heading text-4xl text-ink">Default Risk Detail: #INT-99283</h2>
        </div>
        <div className="flex space-x-gutter">
          <button className="flex items-center gap-2 border-[1.5px] border-copper text-copper px-4 py-2 rounded font-mono text-xs font-bold hover:bg-copper hover:text-white transition-all uppercase tracking-widest">
            <Download size={18} />
            EXPORT REPORT
          </button>
          <button className="bg-copper text-white px-6 py-2 rounded font-mono text-xs font-bold hover:opacity-90 transition-all flex items-center gap-2 shadow-ambient uppercase tracking-widest">
            <History size={18} />
            COMPARE VERSIONS
          </button>
        </div>
      </div>

      {/* Dashboard Layout: Bento Grid Style */}
      <div className="grid grid-cols-12 gap-gutter">
        {/* Risk Gauge Card */}
        <div className="col-span-12 lg:col-span-7 bg-white rounded-lg border border-rule shadow-ambient p-8 relative overflow-hidden">
          <div className="absolute top-0 left-0 w-full h-1 bg-slate-blue"></div>
          <div className="flex flex-col items-center justify-center py-stack-md">
            <div className="text-center mb-6">
              <span className="font-mono text-[13px] text-slate-500 uppercase tracking-widest">Probability of Default</span>
              <div className="flex items-center justify-center mt-2">
                <span className="bg-sage/10 text-sage px-3 py-1 rounded-full font-mono text-[11px] border border-sage/20 uppercase tracking-wider font-bold">Low Risk</span>
              </div>
            </div>
            
            {/* Gauge */}
            <div className="relative w-72 h-36 overflow-hidden">
              <svg className="w-full h-72 transform -rotate-180" viewBox="0 0 100 50">
                <path 
                  d="M 10 50 A 40 40 0 0 1 90 50" 
                  fill="none" 
                  stroke="#E9D6CF" 
                  strokeLinecap="round" 
                  strokeWidth="8"
                />
                <motion.path 
                  initial={{ strokeDashoffset: 126 }}
                  animate={{ strokeDashoffset: 106 }} // ~15%
                  d="M 10 50 A 40 40 0 0 1 90 50" 
                  fill="none" 
                  stroke="#4A7C59" 
                  strokeDasharray="126" 
                  strokeLinecap="round" 
                  strokeWidth="8"
                />
              </svg>
              {/* Needle */}
              <motion.div 
                initial={{ rotate: -90 }}
                animate={{ rotate: -63 }} // -90 to 90 scale, 15% is approx -63
                className="absolute bottom-0 left-1/2 w-1 h-32 bg-ink origin-bottom -translate-x-1/2 transition-transform duration-1000 ease-out"
              >
                <div className="w-4 h-4 bg-ink rounded-full absolute bottom-[-8px] left-[-6px]"></div>
              </motion.div>
            </div>

            <div className="mt-8 flex flex-col items-center">
              <span className="font-mono text-7xl font-bold leading-none text-ink">15%</span>
              <p className="text-body text-slate-500 mt-4 max-w-xs text-center italic">The predictive model estimates a high likelihood of repayment based on current capital liquidity and historical bill ratios.</p>
            </div>
          </div>

          {/* Risk Legend */}
          <div className="mt-8 border-t border-rule pt-6 flex justify-between items-center px-12">
            {[
              { label: 'LOW (0-30%)', color: 'bg-sage' },
              { label: 'MEDIUM (31-70%)', color: 'bg-amber' },
              { label: 'HIGH (71-100%)', color: 'bg-copper' },
            ].map((item, idx) => (
              <div key={idx} className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${item.color}`}></div>
                <span className="font-mono text-[11px] text-slate-500 tracking-tight uppercase">{item.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Column 2 */}
        <div className="col-span-12 lg:col-span-5 space-y-gutter">
          {/* Parameters Used */}
          <div className="bg-white rounded-lg border border-rule shadow-ambient p-6">
            <h3 className="font-mono text-xs font-bold text-ink uppercase mb-4 border-b border-rule pb-2 flex items-center gap-2 tracking-widest">
              <Activity size={16} />
              Parameters Used
            </h3>
            <div className="grid grid-cols-2 gap-y-4">
              {[
                { label: 'Age of Entity', value: '25 Years' },
                { label: 'Credit Limit', value: '$100,000' },
                { label: 'Marital Status', value: 'Married' },
                { label: 'Education Level', value: 'Graduate' },
              ].map((p, idx) => (
                <div key={idx}>
                  <span className="text-mono text-[10px] text-slate-400 block uppercase tracking-wide">{p.label}</span>
                  <span className="font-mono text-sm text-ink font-bold">{p.value}</span>
                </div>
              ))}
              <div className="col-span-2 bg-mist p-3 rounded border border-rule/50">
                <span className="font-mono text-[10px] text-slate-400 block mb-1 uppercase tracking-wide">Recent Credit Inquiry</span>
                <div className="flex justify-between items-center">
                  <span className="font-mono text-sm text-sage font-bold">None detected</span>
                  <CheckCircle2 className="text-sage" size={18} />
                </div>
              </div>
            </div>
          </div>

          {/* Risk Factors */}
          <div className="bg-white rounded-lg border border-rule shadow-ambient p-6">
            <h3 className="font-mono text-xs font-bold text-ink uppercase mb-4 border-b border-rule pb-2 flex items-center gap-2 tracking-widest">
              <AlertTriangle size={16} />
              Risk Factors
            </h3>
            <ul className="space-y-3">
              <li className="flex items-start gap-4 p-3 bg-red-50 rounded-lg border border-red-100">
                <TrendingDown className="text-error shrink-0" size={20} />
                <div>
                  <span className="block font-heading font-bold text-ink text-sm">Low pay/bill ratio</span>
                  <p className="text-[11px] text-slate-500 mt-0.5 leading-normal">Payment activity has decreased by 12% over the last fiscal quarter.</p>
                </div>
              </li>
              <li className="flex items-start gap-4 p-3 bg-sage/10 rounded-lg border border-sage/20">
                <Verified className="text-sage shrink-0" size={20} />
                <div>
                  <span className="block font-heading font-bold text-ink text-sm">Consistent Liquidity</span>
                  <p className="text-[11px] text-slate-500 mt-0.5 leading-normal">Cash reserves have maintained a 3:1 ratio relative to current debt obligations.</p>
                </div>
              </li>
            </ul>
          </div>
        </div>

        {/* Secondary Data Visualization Section */}
        <div className="col-span-12 bg-white rounded-lg border border-rule shadow-ambient p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="font-mono text-xs font-bold text-ink uppercase tracking-widest">Historical Prediction Context</h3>
            <div className="flex items-center gap-4">
              <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest">View Mode:</span>
              <div className="flex bg-mist rounded p-1 border border-rule">
                <button className="px-3 py-1 bg-white text-ink shadow-sm rounded font-mono text-[10px] font-bold">GRID</button>
                <button className="px-3 py-1 text-slate-400 hover:text-ink transition-colors font-mono text-[10px] font-bold">LIST</button>
              </div>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-gutter">
            {[
              { label: 'Market Confidence', value: '92.4%', color: 'bg-sage', icon: TrendingUp },
              { label: 'Model Variance', value: '1.2%', color: 'bg-amber', icon: Activity },
              { label: 'Peer Benchmark', value: 'Top 5%', color: 'bg-copper', icon: Verified },
            ].map((stat, idx) => (
              <div key={idx} className="p-4 border border-rule rounded hover:border-copper transition-colors group cursor-default">
                <div className="flex justify-between items-center mb-3">
                  <span className="font-mono text-[10px] text-slate-500 uppercase tracking-widest">{stat.label}</span>
                  <stat.icon className={`${stat.color.replace('bg-', 'text-')} group-hover:scale-110 transition-transform`} size={16} />
                </div>
                <div className="text-2xl font-mono text-ink font-bold">{stat.value}</div>
                <div className="mt-4 h-1 w-full bg-mist overflow-hidden rounded">
                  <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: idx === 0 ? '92%' : idx === 1 ? '15%' : '95%' }}
                    className={`${stat.color} h-full`}
                  ></motion.div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Detailed Table */}
        <div className="col-span-12">
          <div className="bg-white rounded-xl border border-rule shadow-ambient overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-blue bg-slate-50/50">
              <h3 className="font-mono text-xs font-bold text-slate-blue uppercase tracking-widest">Bill Payment History (Raw Data)</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead className="bg-mist/30 border-b border-rule">
                  <tr>
                    <th className="px-6 py-4 font-mono text-[10px] text-slate-500 uppercase tracking-widest">Period</th>
                    <th className="px-6 py-4 font-mono text-[10px] text-slate-500 uppercase tracking-widest">Bill Amount</th>
                    <th className="px-6 py-4 font-mono text-[10px] text-slate-500 uppercase tracking-widest">Paid Amount</th>
                    <th className="px-6 py-4 font-mono text-[10px] text-slate-500 uppercase tracking-widest">Status</th>
                    <th className="px-6 py-4 font-mono text-[10px] text-slate-500 uppercase tracking-widest text-right">Variance</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-rule font-body text-sm">
                  {[
                    { period: 'SEP 2023', bill: '$12,400.00', paid: '$12,400.00', status: 'Settled', variance: '0.0%', color: 'text-sage' },
                    { period: 'AUG 2023', bill: '$11,200.00', paid: '$10,000.00', status: 'Partial', variance: '-10.7%', color: 'text-amber' },
                    { period: 'JUL 2023', bill: '$14,500.00', paid: '$14,500.00', status: 'Settled', variance: '0.0%', color: 'text-sage' },
                  ].map((row, idx) => (
                    <tr key={idx} className="hover:bg-mist/20 transition-colors">
                      <td className="px-6 py-4 font-mono text-ink font-bold">{row.period}</td>
                      <td className="px-6 py-4 text-slate-600">{row.bill}</td>
                      <td className="px-6 py-4 text-slate-600">{row.paid}</td>
                      <td className="px-6 py-4">
                        <span className={`text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full border uppercase tracking-wider ${
                          row.status === 'Settled' ? 'bg-sage/10 text-sage border-sage/20' : 'bg-amber/10 text-amber border-amber/20'
                        }`}>
                          {row.status}
                        </span>
                      </td>
                      <td className={`px-6 py-4 font-mono font-bold text-right ${row.color}`}>{row.variance}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
