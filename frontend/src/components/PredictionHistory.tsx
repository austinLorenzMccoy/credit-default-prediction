import React from 'react';
import { Download, Search, Filter, ChevronLeft, ChevronRight, BarChart3, CheckCircle2, History, ShieldCheck, Activity, TrendingUp } from 'lucide-react';
import { Prediction } from '../types';
import { motion } from 'motion/react';

const mockPredictions: Prediction[] = [
  { id: 'GTS-4492-B', date: 'Oct 24, 2023', entityName: 'Global Tech Solutions', type: 'CREDIT LIMIT', riskScore: 12.4, limit: 450000, status: 'APPROVED' },
  { id: 'RLX-8812-A', date: 'Oct 23, 2023', entityName: 'Redwood Logistics', type: 'DEFAULT RISK', riskScore: 78.2, status: 'HIGH RISK' },
  { id: 'ALP-0023-F', date: 'Oct 22, 2023', entityName: 'Alpine Ventures', type: 'CREDIT LIMIT', riskScore: 44.1, limit: 125000, status: 'RE-EVALUATE' },
  { id: 'ICG-1209-C', date: 'Oct 22, 2023', entityName: 'Inland Capital Group', type: 'CREDIT LIMIT', riskScore: 8.5, limit: 2500000, status: 'APPROVED' },
  { id: 'ZPH-3341-X', date: 'Oct 21, 2023', entityName: 'Zion Pharma', type: 'DEFAULT RISK', riskScore: 32.9, status: 'MODERATE' },
];

export function PredictionHistory() {
  return (
    <div className="space-y-stack-md">
      {/* Page Header */}
      <div className="flex justify-between items-end mb-stack-md">
        <div>
          <h2 className="font-heading text-4xl text-ink mb-2">Prediction History</h2>
          <p className="text-body text-slate-500">A comprehensive log of all risk assessments and credit limit simulations.</p>
        </div>
        <button className="flex items-center space-x-2 px-5 py-2.5 bg-white border border-rule rounded-xl text-ink font-mono text-xs font-bold hover:bg-mist transition-all shadow-sm uppercase tracking-widest">
          <Download size={18} />
          <span>Export CSV</span>
        </button>
      </div>

      {/* Filters */}
      <div className="grid grid-cols-12 gap-gutter mb-stack-md">
        <div className="col-span-12 lg:col-span-9 bg-white p-5 rounded-xl border border-rule shadow-ambient flex items-center space-x-6">
          <div className="flex-1">
            <label className="font-mono text-[11px] text-slate-blue mb-2 block uppercase tracking-widest font-bold">Search by Client</label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
              <input 
                className="w-full pl-10 pr-4 py-2 bg-mist border border-rule rounded-lg focus:ring-2 focus:ring-copper focus:border-copper outline-none transition-all font-body text-sm" 
                placeholder="ID or Entity name..." 
                type="text"
              />
            </div>
          </div>
          <div>
            <label className="font-mono text-[11px] text-slate-blue mb-2 block uppercase tracking-widest font-bold">Type</label>
            <select className="w-48 py-2 px-3 bg-mist border border-rule rounded-lg focus:ring-2 focus:ring-copper focus:border-copper outline-none font-body text-sm cursor-pointer appearance-none">
              <option>All Types</option>
              <option>Default Risk</option>
              <option>Credit Limit</option>
            </select>
          </div>
          <div>
            <label className="font-mono text-[11px] text-slate-blue mb-2 block uppercase tracking-widest font-bold">Risk Level</label>
            <select className="w-48 py-2 px-3 bg-mist border border-rule rounded-lg focus:ring-2 focus:ring-copper focus:border-copper outline-none font-body text-sm cursor-pointer appearance-none">
              <option>All Risks</option>
              <option>Low Risk</option>
              <option>Moderate Risk</option>
              <option>High Risk</option>
            </select>
          </div>
          <div className="pt-6">
            <button className="p-2.5 text-copper hover:bg-mist rounded-lg transition-colors border border-copper/20">
              <Filter size={20} />
            </button>
          </div>
        </div>
        <div className="col-span-12 lg:col-span-3 bg-slate-blue text-white p-6 rounded-xl shadow-lg flex flex-col justify-center">
          <p className="text-[10px] font-mono uppercase tracking-widest opacity-60 mb-1 font-bold">Total Predictions</p>
          <p className="font-mono text-3xl font-black">1,482</p>
        </div>
      </div>

      {/* Data Table */}
      <div className="bg-white rounded-xl border border-rule shadow-ambient overflow-hidden mb-stack-md">
        <div className="h-1.5 w-full bg-slate-blue"></div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead className="bg-mist/30 border-b border-rule">
              <tr>
                <th className="px-6 py-4 font-mono text-[10px] text-slate-blue uppercase tracking-widest font-bold">DATE</th>
                <th className="px-6 py-4 font-mono text-[10px] text-slate-blue uppercase tracking-widest font-bold">ENTITY / ID</th>
                <th className="px-6 py-4 font-mono text-[10px] text-slate-blue uppercase tracking-widest font-bold">TYPE</th>
                <th className="px-6 py-4 font-mono text-[10px] text-slate-blue uppercase tracking-widest font-bold text-right">RISK %</th>
                <th className="px-6 py-4 font-mono text-[10px] text-slate-blue uppercase tracking-widest font-bold text-right">LIMIT</th>
                <th className="px-6 py-4 font-mono text-[10px] text-slate-blue uppercase tracking-widest font-bold">STATUS</th>
                <th className="px-6 py-4 font-mono text-[10px] text-slate-blue uppercase tracking-widest font-bold text-center">ACTIONS</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-rule">
              {mockPredictions.map((pred, idx) => (
                <tr key={idx} className="hover:bg-mist/20 transition-colors group">
                  <td className="px-6 py-4 font-body text-sm text-slate-600">{pred.date}</td>
                  <td className="px-6 py-4">
                    <div className="font-heading font-black text-ink text-sm uppercase tracking-tight">{pred.entityName}</div>
                    <div className="text-[9px] font-mono text-slate-400 uppercase tracking-widest mt-0.5">ID: {pred.id}</div>
                  </td>
                  <td className="px-6 py-4">
                    <span className="px-2 py-0.5 bg-slate-100 text-slate-600 text-[9px] font-mono font-bold rounded uppercase tracking-wider border border-slate-200">
                      {pred.type}
                    </span>
                  </td>
                  <td className={`px-6 py-4 text-right font-mono text-lg font-bold ${
                    pred.riskScore < 20 ? 'text-sage' : pred.riskScore < 50 ? 'text-amber' : 'text-copper'
                  }`}>
                    {pred.riskScore}%
                  </td>
                  <td className="px-6 py-4 text-right font-mono text-sm font-bold text-slate-700">
                    {pred.limit ? `$${pred.limit.toLocaleString()}` : '—'}
                  </td>
                  <td className="px-6 py-4">
                    <span className={`inline-flex items-center px-4 py-0.5 rounded-full text-[9px] font-black border uppercase tracking-widest ${
                      pred.status === 'APPROVED' ? 'bg-sage/10 text-sage border-sage/20' :
                      pred.status === 'HIGH RISK' ? 'bg-red-50 text-error border-red-200' :
                      'bg-amber/10 text-amber border-amber/20'
                    }`}>
                      {pred.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-center">
                    <button className="text-copper hover:underline text-[11px] font-mono font-bold uppercase tracking-widest hover:scale-105 transition-transform">
                      View Report
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        {/* Pagination */}
        <div className="bg-mist/30 px-6 py-5 border-t border-rule flex items-center justify-between">
          <p className="font-mono text-[10px] text-slate-500 uppercase tracking-widest font-medium">Showing 1 to 5 of 1,482 entries</p>
          <div className="flex items-center space-x-1.5">
            <button className="p-1.5 border border-rule rounded bg-white text-slate-400 hover:text-copper disabled:opacity-30 disabled:hover:text-slate-400" disabled>
              <ChevronLeft size={16} />
            </button>
            <button className="px-4 py-1.5 border border-copper bg-copper text-white font-mono text-xs font-bold rounded shadow-sm">1</button>
            <button className="px-4 py-1.5 border border-rule bg-white text-slate-500 font-mono text-xs font-bold rounded hover:bg-mist transition-colors">2</button>
            <button className="px-4 py-1.5 border border-rule bg-white text-slate-500 font-mono text-xs font-bold rounded hover:bg-mist transition-colors">3</button>
            <span className="px-2 text-slate-400 font-bold">...</span>
            <button className="px-4 py-1.5 border border-rule bg-white text-slate-500 font-mono text-xs font-bold rounded hover:bg-mist transition-colors">297</button>
            <button className="p-1.5 border border-rule rounded bg-white text-slate-400 hover:text-copper transition-colors">
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Insights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-gutter">
        <div className="bg-white border border-rule p-6 rounded-xl shadow-ambient group">
          <div className="flex items-center space-x-3 mb-6">
            <TrendingUp className="text-copper group-hover:scale-110 transition-transform" size={20} />
            <h3 className="font-mono text-xs font-bold text-ink uppercase tracking-widest">Volume Trend</h3>
          </div>
          <div className="h-24 flex items-end space-x-2">
            {[0.5, 0.7, 0.4, 1, 0.8, 0.6].map((h, i) => (
              <motion.div 
                key={i}
                initial={{ height: 0 }}
                animate={{ height: `${h * 100}%` }}
                className={`w-full ${i === 3 ? 'bg-copper' : 'bg-mist'} rounded-t-sm transition-colors group-hover:bg-opacity-80`}
              ></motion.div>
            ))}
          </div>
          <p className="mt-4 text-xs font-body text-slate-400 flex justify-between">
            <span>+14% vs previous week</span>
            <span className="font-mono font-bold text-copper">-</span>
          </p>
        </div>
        
        <div className="bg-white border border-rule p-6 rounded-xl shadow-ambient group">
          <div className="flex items-center space-x-3 mb-6">
            <ShieldCheck className="text-sage group-hover:scale-110 transition-transform" size={20} />
            <h3 className="font-mono text-xs font-bold text-ink uppercase tracking-widest">Approval Rate</h3>
          </div>
          <p className="font-mono text-4xl font-black text-sage tracking-tighter">68.2%</p>
          <p className="mt-3 text-xs font-body text-slate-400 leading-relaxed uppercase tracking-tight">Consistent with Q3 target performance threshold.</p>
        </div>

        <div className="bg-white border border-rule p-6 rounded-xl shadow-ambient group">
          <div className="flex items-center space-x-3 mb-6">
            <Activity className="text-slate-blue group-hover:scale-110 transition-transform" size={20} />
            <h3 className="font-mono text-xs font-bold text-ink uppercase tracking-widest">Avg Response</h3>
          </div>
          <p className="font-mono text-4xl font-black text-slate-blue tracking-tighter">1.2s</p>
          <p className="mt-3 text-xs font-body text-slate-400 leading-relaxed uppercase tracking-tight">Real-time inference performance benchmarked across nodes.</p>
        </div>
      </div>

      {/* Footer Branding */}
      <footer className="mt-20 p-8 border-t border-rule flex justify-between items-center text-slate-400 opacity-60">
        <p className="text-[10px] font-mono uppercase tracking-widest font-bold">© 2026 CreditIntel Pro • Proprietary Algorithm v4.2.1</p>
        <div className="flex space-x-6">
          <ShieldCheck size={20} />
          <CheckCircle2 size={20} />
        </div>
      </footer>
    </div>
  );
}
