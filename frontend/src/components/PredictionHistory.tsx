import React, { useState, useMemo } from 'react';
import { Download, Search, Filter, ChevronLeft, ChevronRight, ShieldCheck, CheckCircle2, Activity, TrendingUp } from 'lucide-react';
import { Prediction } from '../types';
import { motion } from 'motion/react';

const ALL_PREDICTIONS: Prediction[] = [
  { id: 'GTS-4492-B', date: 'Oct 24, 2023', entityName: 'Global Tech Solutions',   type: 'CREDIT LIMIT',  riskScore: 12.4, limit: 450000,   status: 'APPROVED' },
  { id: 'RLX-8812-A', date: 'Oct 23, 2023', entityName: 'Redwood Logistics',        type: 'DEFAULT RISK',  riskScore: 78.2,              status: 'HIGH RISK' },
  { id: 'ALP-0023-F', date: 'Oct 22, 2023', entityName: 'Alpine Ventures',          type: 'CREDIT LIMIT',  riskScore: 44.1, limit: 125000, status: 'RE-EVALUATE' },
  { id: 'ICG-1209-C', date: 'Oct 22, 2023', entityName: 'Inland Capital Group',     type: 'CREDIT LIMIT',  riskScore: 8.5,  limit: 2500000,status: 'APPROVED' },
  { id: 'ZPH-3341-X', date: 'Oct 21, 2023', entityName: 'Zion Pharma',              type: 'DEFAULT RISK',  riskScore: 32.9,              status: 'MODERATE' },
  { id: 'FNB-7723-D', date: 'Oct 20, 2023', entityName: 'First National Brokers',   type: 'DEFAULT RISK',  riskScore: 19.1,              status: 'APPROVED' },
  { id: 'SLK-5514-G', date: 'Oct 19, 2023', entityName: 'Silk Road Exports',        type: 'CREDIT LIMIT',  riskScore: 61.3, limit: 80000,  status: 'RE-EVALUATE' },
  { id: 'OBJ-2200-K', date: 'Oct 18, 2023', entityName: 'Objective Capital',        type: 'DEFAULT RISK',  riskScore: 84.7,              status: 'HIGH RISK' },
];

const PAGE_SIZE = 5;

const statusStyle: Record<string, string> = {
  'APPROVED':    'bg-sage/10 text-sage border-sage/20',
  'HIGH RISK':   'bg-red-50 text-error border-red-200',
  'RE-EVALUATE': 'bg-amber/10 text-amber border-amber/20',
  'MODERATE':    'bg-amber/10 text-amber border-amber/20',
  'SETTLED':     'bg-sage/10 text-sage border-sage/20',
  'PARTIAL':     'bg-amber/10 text-amber border-amber/20',
};

export function PredictionHistory() {
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState('All Types');
  const [riskFilter, setRiskFilter] = useState('All Risks');
  const [page, setPage] = useState(1);

  const filtered = useMemo(() => {
    return ALL_PREDICTIONS.filter(p => {
      const matchSearch = search === '' ||
        p.entityName.toLowerCase().includes(search.toLowerCase()) ||
        p.id.toLowerCase().includes(search.toLowerCase());
      const matchType = typeFilter === 'All Types' || p.type === typeFilter.toUpperCase().replace(' ', ' ');
      const matchRisk = riskFilter === 'All Risks' ||
        (riskFilter === 'Low Risk' && p.riskScore < 30) ||
        (riskFilter === 'Moderate Risk' && p.riskScore >= 30 && p.riskScore < 70) ||
        (riskFilter === 'High Risk' && p.riskScore >= 70);
      return matchSearch && matchType && matchRisk;
    });
  }, [search, typeFilter, riskFilter]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  const handleFilter = () => setPage(1);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex justify-between items-end">
        <div>
          <h2 className="font-heading text-4xl text-ink mb-2">Prediction History</h2>
          <p className="font-body text-slate-500">A comprehensive log of all risk assessments and credit limit simulations.</p>
        </div>
        <button className="flex items-center gap-2 px-5 py-2.5 bg-white border border-rule rounded-xl font-mono text-[10px] font-bold uppercase tracking-widest text-ink hover:bg-mist transition-all shadow-ambient">
          <Download size={16} /> Export CSV
        </button>
      </div>

      {/* Filters */}
      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-12 lg:col-span-9 bg-white p-5 rounded-xl border border-rule shadow-ambient flex flex-wrap items-end gap-5">
          <div className="flex-1 min-w-[180px]">
            <label className="font-mono text-[10px] text-slate-blue mb-1.5 block uppercase tracking-widest font-bold">Search</label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
              <input
                className="w-full pl-9 pr-4 h-10 bg-mist border border-rule rounded-lg font-body text-sm focus:outline-none focus:ring-2 focus:ring-copper"
                placeholder="ID or entity name…"
                value={search}
                onChange={e => { setSearch(e.target.value); setPage(1); }}
              />
            </div>
          </div>
          <div>
            <label className="font-mono text-[10px] text-slate-blue mb-1.5 block uppercase tracking-widest font-bold">Type</label>
            <select
              className="h-10 px-3 bg-mist border border-rule rounded-lg font-body text-sm focus:outline-none focus:ring-2 focus:ring-copper appearance-none cursor-pointer"
              value={typeFilter}
              onChange={e => { setTypeFilter(e.target.value); handleFilter(); }}
            >
              <option>All Types</option>
              <option>Default Risk</option>
              <option>Credit Limit</option>
            </select>
          </div>
          <div>
            <label className="font-mono text-[10px] text-slate-blue mb-1.5 block uppercase tracking-widest font-bold">Risk Level</label>
            <select
              className="h-10 px-3 bg-mist border border-rule rounded-lg font-body text-sm focus:outline-none focus:ring-2 focus:ring-copper appearance-none cursor-pointer"
              value={riskFilter}
              onChange={e => { setRiskFilter(e.target.value); handleFilter(); }}
            >
              <option>All Risks</option>
              <option>Low Risk</option>
              <option>Moderate Risk</option>
              <option>High Risk</option>
            </select>
          </div>
          <button className="h-10 px-3 border border-copper/30 text-copper hover:bg-mist rounded-lg transition-colors">
            <Filter size={18} />
          </button>
        </div>

        <div className="col-span-12 lg:col-span-3 bg-slate-blue text-white p-6 rounded-xl shadow-lifted flex flex-col justify-center">
          <p className="font-mono text-[9px] uppercase tracking-widest text-white/50 mb-1 font-bold">Filtered Results</p>
          <p className="font-mono text-3xl font-bold">{filtered.length.toLocaleString()}</p>
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl border border-rule shadow-ambient overflow-hidden">
        <div className="h-1 w-full bg-slate-blue" />
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead className="bg-mist/40 border-b border-rule">
              <tr>
                {['Date', 'Entity / ID', 'Type', 'Risk %', 'Limit', 'Status', 'Actions'].map((h, i) => (
                  <th key={h} className={`px-6 py-4 font-mono text-[10px] text-slate-blue uppercase tracking-widest font-bold ${i >= 3 ? 'text-right' : ''} ${i === 6 ? 'text-center' : ''}`}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-rule">
              {paginated.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-6 py-12 text-center font-body text-slate-400 text-sm">
                    No predictions match your filters.
                  </td>
                </tr>
              ) : paginated.map((pred, idx) => (
                <tr key={pred.id} className="hover:bg-mist/20 transition-colors">
                  <td className="px-6 py-4 font-body text-sm text-slate-600">{pred.date}</td>
                  <td className="px-6 py-4">
                    <div className="font-heading font-bold text-ink text-sm">{pred.entityName}</div>
                    <div className="font-mono text-[9px] text-slate-400 uppercase tracking-widest mt-0.5">ID: {pred.id}</div>
                  </td>
                  <td className="px-6 py-4">
                    <span className="px-2 py-0.5 bg-slate-100 text-slate-600 text-[9px] font-mono font-bold rounded uppercase tracking-wider border border-slate-200">
                      {pred.type}
                    </span>
                  </td>
                  <td className={`px-6 py-4 text-right font-mono text-lg font-bold ${
                    pred.riskScore < 30 ? 'text-sage' : pred.riskScore < 70 ? 'text-amber' : 'text-copper'
                  }`}>
                    {pred.riskScore.toFixed(1)}%
                  </td>
                  <td className="px-6 py-4 text-right font-mono text-sm font-bold text-slate-700">
                    {pred.limit ? `₦${pred.limit.toLocaleString()}` : '—'}
                  </td>
                  <td className="px-6 py-4">
                    <span className={`inline-flex items-center px-3 py-0.5 rounded-full text-[9px] font-bold border uppercase tracking-widest ${statusStyle[pred.status] ?? ''}`}>
                      {pred.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-center">
                    <button className="font-mono text-[10px] text-copper hover:underline font-bold uppercase tracking-widest">
                      View Report
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="bg-mist/30 px-6 py-4 border-t border-rule flex items-center justify-between">
          <p className="font-mono text-[10px] text-slate-500 uppercase tracking-widest">
            Showing {Math.min((page - 1) * PAGE_SIZE + 1, filtered.length)}–{Math.min(page * PAGE_SIZE, filtered.length)} of {filtered.length}
          </p>
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
              className="p-1.5 border border-rule rounded bg-white text-slate-400 hover:text-copper disabled:opacity-30 transition-colors"
            >
              <ChevronLeft size={15} />
            </button>
            {Array.from({ length: totalPages }, (_, i) => i + 1).map(n => (
              <button
                key={n}
                onClick={() => setPage(n)}
                className={`px-3.5 py-1.5 border rounded font-mono text-xs font-bold transition-colors ${
                  n === page
                    ? 'border-copper bg-copper text-white shadow-ambient'
                    : 'border-rule bg-white text-slate-500 hover:bg-mist'
                }`}
              >
                {n}
              </button>
            ))}
            <button
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="p-1.5 border border-rule rounded bg-white text-slate-400 hover:text-copper disabled:opacity-30 transition-colors"
            >
              <ChevronRight size={15} />
            </button>
          </div>
        </div>
      </div>

      {/* Insight cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[
          { icon: TrendingUp, color: 'text-copper', label: 'Volume Trend', value: '+14%', sub: 'vs previous week', bars: [0.5,0.7,0.4,1,0.8,0.6] },
          { icon: ShieldCheck, color: 'text-sage',   label: 'Approval Rate', value: '68.2%', sub: 'Consistent with Q3 target' },
          { icon: Activity,   color: 'text-slate-blue', label: 'Avg Response', value: '1.2s', sub: 'Real-time inference performance' },
        ].map(({ icon: Icon, color, label, value, sub, bars }) => (
          <div key={label} className="bg-white border border-rule p-6 rounded-xl shadow-ambient">
            <div className="flex items-center gap-3 mb-5">
              <Icon className={color} size={18} />
              <h3 className="font-mono text-[10px] font-bold text-ink uppercase tracking-widest">{label}</h3>
            </div>
            {bars ? (
              <div className="h-16 flex items-end gap-1.5 mb-3">
                {bars.map((h, i) => (
                  <motion.div key={i} initial={{ height: 0 }} animate={{ height: `${h * 100}%` }}
                    className={`flex-1 rounded-t-sm ${i === 3 ? 'bg-copper' : 'bg-mist'}`} />
                ))}
              </div>
            ) : null}
            <p className={`font-mono text-3xl font-bold tracking-tight ${color}`}>{value}</p>
            <p className="font-body text-xs text-slate-400 mt-2 leading-relaxed">{sub}</p>
          </div>
        ))}
      </div>

      <footer className="mt-8 py-6 border-t border-rule flex justify-between items-center text-slate-300">
        <p className="font-mono text-[9px] uppercase tracking-widest font-bold">© 2026 CreditIntel Pro · Algorithm v4.2.1</p>
        <div className="flex gap-5">
          <ShieldCheck size={18} /><CheckCircle2 size={18} />
        </div>
      </footer>
    </div>
  );
}
