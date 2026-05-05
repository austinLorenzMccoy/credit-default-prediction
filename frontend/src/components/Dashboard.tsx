import React, { useState } from 'react';
import { Activity, ShieldCheck, TrendingUp, ChevronRight, UserCircle2, Loader2 } from 'lucide-react';
import { CustomerProfile, Page } from '../types';
import { motion } from 'motion/react';

// ── Inline SVG Gauge (replaces broken border-trick div) ───────────────────
function SemiGauge({ percent, color }: { percent: number; color: string }) {
  const R = 52;
  const cx = 72, cy = 68;
  const circumference = Math.PI * R; // half circle arc length ≈ 163.4
  const offset = circumference - (percent / 100) * circumference;
  // Needle: -90° = far left (0%), +90° = far right (100%)
  const needleAngle = -90 + (percent / 100) * 180;
  const rad = (needleAngle * Math.PI) / 180;
  const nx = cx + R * Math.cos(rad);
  const ny = cy + R * Math.sin(rad);

  return (
    <svg width="144" height="82" viewBox="0 0 144 82" fill="none">
      {/* Track */}
      <path
        d={`M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`}
        stroke="#E8DDD4"
        strokeWidth="10"
        strokeLinecap="round"
        fill="none"
      />
      {/* Fill arc */}
      <path
        d={`M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`}
        stroke={color}
        strokeWidth="10"
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        fill="none"
        style={{ transition: 'stroke-dashoffset 1s ease' }}
      />
      {/* Needle */}
      <line
        x1={cx}
        y1={cy}
        x2={nx}
        y2={ny}
        stroke="#0D1B2A"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <circle cx={cx} cy={cy} r="5" fill="#0D1B2A" />
    </svg>
  );
}

export function Dashboard({ onPredict }: { onPredict: (p: Page) => void }) {
  const [profile, setProfile] = useState<CustomerProfile>({
    age: 34,
    gender: 'Female',
    education: 'Graduate School',
    maritalStatus: 'Married',
    desiredLimit: 150000,
    payStatus: 'Paid on Time',
    billAmount: 45200,
    paymentAmount: 10000,
  });

  const [isLoading, setIsLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  // Simulate prediction for demo
  const handlePredictDefaultRisk = async () => {
    setIsLoading(true);
    setApiError(null);
    await new Promise(r => setTimeout(r, 900));
    setIsLoading(false);
    onPredict('default-detail');
  };

  const handlePredictCreditLimit = async () => {
    setIsLoading(true);
    setApiError(null);
    await new Promise(r => setTimeout(r, 900));
    setIsLoading(false);
    onPredict('limit-detail');
  };

  const field = (label: string, children: React.ReactNode) => (
    <div>
      <label className="block font-mono text-[11px] text-slate-blue uppercase tracking-widest mb-1.5 font-bold">
        {label}
      </label>
      {children}
    </div>
  );

  const inputCls = "w-full h-11 border border-rule rounded-lg px-4 bg-white font-body text-sm text-ink focus:outline-none focus:ring-2 focus:ring-copper focus:ring-offset-1 transition-all";
  const selectCls = inputCls + " cursor-pointer appearance-none";

  return (
    <div className="space-y-8">
      <header className="mb-10">
        <h2 className="font-heading text-4xl text-ink mb-2">Customer Intelligence</h2>
        <p className="font-body text-slate-blue/70">Execute precision risk modelling for high-value credit applications.</p>
      </header>

      <div className="grid grid-cols-12 gap-8">
        {/* ── INPUT FORM ────────────────────────────────────────── */}
        <section className="col-span-12 lg:col-span-8">
          <div className="bg-white rounded-xl border border-rule border-t-4 border-t-slate-blue p-8 shadow-ambient">
            <div className="flex items-center justify-between mb-8 border-b border-rule pb-4">
              <h3 className="font-mono text-xs font-bold text-ink flex items-center gap-2 uppercase tracking-widest">
                <Activity className="text-copper" size={18} />
                Customer Financial Profile
              </h3>
              <span className="font-mono text-[10px] text-slate-400 bg-mist px-2 py-1 rounded">SESSION: CI-99284</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Demographics */}
              <div className="space-y-4">
                {field('Age',
                  <input type="number" className={inputCls} value={profile.age}
                    onChange={e => setProfile({ ...profile, age: parseInt(e.target.value) || 18 })} />
                )}
                {field('Gender',
                  <select className={selectCls} value={profile.gender}
                    onChange={e => setProfile({ ...profile, gender: e.target.value as any })}>
                    <option>Female</option><option>Male</option><option>Other</option>
                  </select>
                )}
                {field('Education',
                  <select className={selectCls} value={profile.education}
                    onChange={e => setProfile({ ...profile, education: e.target.value as any })}>
                    <option>Graduate School</option><option>University</option><option>High School</option>
                  </select>
                )}
                {field('Marital Status',
                  <select className={selectCls} value={profile.maritalStatus}
                    onChange={e => setProfile({ ...profile, maritalStatus: e.target.value as any })}>
                    <option>Married</option><option>Single</option><option>Divorced</option>
                  </select>
                )}
              </div>

              {/* Financials */}
              <div className="space-y-4">
                {field('Desired Credit Limit (₦)',
                  <div className="relative">
                    <span className="absolute left-4 top-1/2 -translate-y-1/2 font-mono text-[13px] text-slate-400">₦</span>
                    <input type="text" className={inputCls + " pl-8"} 
                      value={profile.desiredLimit.toLocaleString()}
                      onChange={e => setProfile({ ...profile, desiredLimit: parseInt(e.target.value.replace(/,/g, '')) || 0 })} />
                  </div>
                )}
                {field('Pay Status (Last Month)',
                  <select className={selectCls} value={profile.payStatus}
                    onChange={e => setProfile({ ...profile, payStatus: e.target.value as any })}>
                    <option>Paid on Time</option><option>1 Month Delay</option><option>2 Month Delay</option>
                  </select>
                )}
                {field('Bill Amount (₦)',
                  <div className="relative">
                    <span className="absolute left-4 top-1/2 -translate-y-1/2 font-mono text-[13px] text-slate-400">₦</span>
                    <input type="text" className={inputCls + " pl-8"}
                      value={profile.billAmount.toLocaleString()}
                      onChange={e => setProfile({ ...profile, billAmount: parseInt(e.target.value.replace(/,/g, '')) || 0 })} />
                  </div>
                )}
                {field('Payment Amount (₦)',
                  <div className="relative">
                    <span className="absolute left-4 top-1/2 -translate-y-1/2 font-mono text-[13px] text-slate-400">₦</span>
                    <input type="text" className={inputCls + " pl-8"}
                      value={profile.paymentAmount.toLocaleString()}
                      onChange={e => setProfile({ ...profile, paymentAmount: parseInt(e.target.value.replace(/,/g, '')) || 0 })} />
                  </div>
                )}
              </div>
            </div>

            {apiError && (
              <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-error text-sm font-mono">{apiError}</p>
              </div>
            )}

            <div className="pt-6 mt-6 border-t border-rule flex justify-end gap-4">
              <button
                onClick={handlePredictCreditLimit}
                disabled={isLoading}
                className="px-6 h-11 border-[1.5px] border-slate-blue text-slate-blue font-mono text-[11px] font-bold uppercase tracking-widest rounded-lg hover:bg-mist transition-colors disabled:opacity-50"
              >
                Predict Credit Limit
              </button>
              <button
                onClick={handlePredictDefaultRisk}
                disabled={isLoading}
                className="px-8 h-11 bg-copper text-white font-mono text-[11px] font-bold uppercase tracking-widest rounded-lg hover:bg-copper-dk shadow-ambient transition-all active:scale-95 disabled:opacity-50 flex items-center gap-2"
              >
                {isLoading
                  ? <><Loader2 className="animate-spin" size={15} /> Processing…</>
                  : 'Calculate Default Risk'}
              </button>
            </div>
          </div>
        </section>

        {/* ── RESULT CARDS ─────────────────────────────────────── */}
        <aside className="col-span-12 lg:col-span-4 space-y-6">
          {/* Default Risk card */}
          <div
            className="bg-white rounded-xl border border-rule p-6 shadow-ambient cursor-pointer card-hover"
            onClick={() => onPredict('default-detail')}
          >
            <div className="flex items-center justify-between mb-4">
              <span className="font-mono text-[10px] uppercase tracking-widest text-slate-blue font-bold">Default Risk</span>
              <span className="px-2.5 py-1 bg-sage/10 text-sage text-[10px] font-bold font-mono rounded-full border border-sage/20 uppercase tracking-wide">
                Low Risk
              </span>
            </div>
            {/* SVG gauge */}
            <div className="flex flex-col items-center mb-4">
              <SemiGauge percent={15} color="#4A7C59" />
              <span className="font-mono text-4xl font-bold text-ink -mt-2 tracking-tight">15%</span>
              <p className="font-body text-xs text-slate-500 text-center mt-2 px-2">
                Probability of default based on payment behaviour.
              </p>
            </div>
            <div className="bg-mist rounded-lg px-4 py-3 flex items-center gap-3">
              <ShieldCheck className="text-sage shrink-0" size={18} />
              <span className="font-mono text-[11px] font-bold text-ink">Strong repayment profile detected.</span>
            </div>
          </div>

          {/* Credit Limit card */}
          <div
            className="bg-white rounded-xl border border-rule border-t-4 border-t-copper p-6 shadow-ambient cursor-pointer card-hover"
            onClick={() => onPredict('limit-detail')}
          >
            <span className="font-mono text-[10px] uppercase tracking-widest text-slate-blue font-bold block mb-4">
              Recommended Limit
            </span>
            <div className="flex items-baseline gap-2 mb-1">
              <span className="font-mono text-3xl text-ink font-bold tracking-tight">₦150,000</span>
              <span className="font-mono text-sage text-sm font-bold flex items-center gap-0.5">
                <TrendingUp size={13} /> +50%
              </span>
            </div>
            <p className="font-body text-xs text-slate-400 italic mb-5">From previous limit: ₦100,000</p>
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-slate-500 uppercase tracking-tight">Liquidity Score</span>
                <span className="font-bold text-ink">88/100</span>
              </div>
              <div className="w-full bg-mist h-2 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: '88%' }}
                  transition={{ duration: 1, ease: 'easeOut' }}
                  className="bg-copper h-full rounded-full"
                />
              </div>
            </div>
          </div>

          {/* Institutional insight */}
          <div className="p-6 rounded-xl bg-slate-blue text-white shadow-lifted relative overflow-hidden group cursor-pointer"
            onClick={() => onPredict('default-detail')}>
            <div className="relative z-10">
              <h4 className="font-heading text-xl text-white mb-2">Institutional Insight</h4>
              <p className="font-body text-sm text-white/75 leading-relaxed mb-4">
                This profile shows exceptional stability in education and payment status. We recommend immediate approval for the requested increase.
              </p>
              <button className="font-mono text-[10px] uppercase tracking-widest flex items-center gap-2 text-sand group-hover:gap-3 transition-all">
                View Detailed Report <ChevronRight size={13} />
              </button>
            </div>
            <UserCircle2 className="absolute -right-10 -bottom-10 text-white/10" size={160} />
          </div>
        </aside>
      </div>

      {/* Footer stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-4">
        {[
          { label: 'Model Version', value: 'TF-CREDIT v4.2' },
          { label: 'Inference Time', value: '142ms' },
          { label: 'Last Profile Scan', value: 'Just Now' },
        ].map((stat, idx) => (
          <div key={idx} className="bg-white border border-rule rounded-xl p-5 flex items-center gap-4">
            <div>
              <p className="font-mono text-[10px] text-slate-500 uppercase tracking-widest">{stat.label}</p>
              <p className="font-mono text-base text-ink font-bold mt-0.5">{stat.value}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
