import React from 'react';
import { TrendingUp, ShieldCheck, Activity, Users, CreditCard, ArrowUpRight, BarChart3 } from 'lucide-react';
import { motion } from 'motion/react';

export function Overview({ onNavigate }: { onNavigate: (page: any) => void }) {
  const stats = [
    { label: 'Active Portfolios', value: '12', icon: CreditCard, color: 'text-copper' },
    { label: 'Avg Risk Score', value: '24.2%', icon: ShieldCheck, color: 'text-sage' },
    { label: 'Total Predictions', value: '1,482', icon: Activity, color: 'text-slate-blue' },
    { label: 'Total Clients', value: '840', icon: Users, color: 'text-amber' },
  ];

  return (
    <div className="space-y-8">
      <header>
        <h2 className="font-display text-4xl text-ink mb-2">Executive Overview</h2>
        <p className="text-body text-slate-blue/70">Real-time credit health monitoring and system benchmarks.</p>
      </header>

      {/* Hero Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, idx) => (
          <motion.div 
            key={idx}
            whileHover={{ y: -4 }}
            className="bg-white p-6 rounded-xl border border-rule shadow-ambient flex items-center gap-4"
          >
            <div className={`p-3 rounded-lg bg-mist ${stat.color}`}>
              <stat.icon size={24} />
            </div>
            <div>
              <p className="font-mono text-[10px] text-slate-500 uppercase tracking-widest">{stat.label}</p>
              <p className="font-mono text-xl font-black text-ink">{stat.value}</p>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Main Charts Row */}
      <div className="grid grid-cols-12 gap-8">
        <div className="col-span-12 lg:col-span-8 bg-white border border-rule rounded-xl p-8 shadow-ambient border-t-2 border-t-slate-blue">
          <div className="flex justify-between items-center mb-10">
            <h3 className="font-mono text-xs font-bold text-ink uppercase tracking-widest">Portfolio Performance</h3>
            <div className="flex gap-2">
              <span className="flex items-center gap-1.5 text-xs font-mono text-sage font-bold">
                <ArrowUpRight size={14} /> +12% YoY
              </span>
            </div>
          </div>
          <div className="h-64 flex items-end gap-3 pb-4">
            {[40, 65, 45, 90, 75, 55, 80, 70, 85, 95, 60, 100].map((h, i) => (
              <motion.div 
                key={i}
                initial={{ height: 0 }}
                animate={{ height: `${h}%` }}
                className={`flex-1 ${i === 11 ? 'bg-copper' : 'bg-mist'} rounded-t-sm hover:bg-slate-200 transition-colors`}
              />
            ))}
          </div>
          <div className="flex justify-between mt-4 font-mono text-[10px] text-slate-400 uppercase">
            <span>Jan</span>
            <span>Jun</span>
            <span>Dec</span>
          </div>
        </div>

        <div className="col-span-12 lg:col-span-4 bg-white border border-rule rounded-xl p-8 shadow-ambient flex flex-col justify-between">
          <div>
            <h3 className="font-mono text-xs font-bold text-ink uppercase tracking-widest mb-6">Prediction Breakdown</h3>
            <div className="space-y-6">
              {[
                { label: 'Approved', value: 68, color: 'bg-sage' },
                { label: 'Re-evaluate', value: 22, color: 'bg-amber' },
                { label: 'Denied', value: 10, color: 'bg-error' },
              ].map((item, idx) => (
                <div key={idx}>
                  <div className="flex justify-between text-xs font-mono mb-2">
                    <span className="text-slate-500 uppercase">{item.label}</span>
                    <span className="font-bold text-ink">{item.value}%</span>
                  </div>
                  <div className="w-full h-1.5 bg-mist rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${item.value}%` }} 
                      className={`${item.color} h-full`}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
          <button 
            onClick={() => onNavigate('history')}
            className="mt-8 border border-copper text-copper font-mono text-[10px] font-bold uppercase tracking-widest py-3 rounded-lg hover:bg-mist transition-colors flex items-center justify-center gap-2"
          >
            View Full Archives <ArrowUpRight size={14} />
          </button>
        </div>
      </div>

      {/* Quick Access Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div 
          onClick={() => onNavigate('profiling')}
          className="group relative bg-ink text-white p-8 rounded-xl overflow-hidden cursor-pointer hover:shadow-xl transition-all"
        >
          <div className="relative z-10">
            <h4 className="font-heading text-2xl mb-2">New Prediction</h4>
            <p className="text-slate-300 text-sm max-w-sm mb-4">Instantly model risk for new applicants using our latest neural architecture.</p>
            <span className="font-mono text-[10px] uppercase font-bold tracking-widest group-hover:translate-x-2 transition-transform flex items-center gap-2">
              Launch Profiler <TrendingUp size={14} />
            </span>
          </div>
          <TrendingUp className="absolute -right-8 -bottom-8 opacity-10 text-white" size={200} />
        </div>

        <div className="bg-slate-50 border border-dashed border-rule p-8 rounded-xl flex items-center justify-center text-center">
          <div>
            <h4 className="font-mono text-[10px] uppercase tracking-widest text-slate-500 mb-2">Next Scheduled Audit</h4>
            <p className="font-heading text-xl text-ink">Global Re-balancing</p>
            <p className="font-mono text-sm text-copper font-bold mt-1">In 14 Days</p>
          </div>
        </div>
      </div>
    </div>
  );
}
