import React from 'react';
import { ArrowRight, CheckCircle2, Zap, Shield, Search, TrendingUp, BarChart3, Database, X, ExternalLink } from 'lucide-react';
import { Page } from '../types';
import { motion, AnimatePresence } from 'motion/react';

// ── Inline SVG dashboard illustration (replaces Unsplash dependency) ───────
function HeroIllustration() {
  return (
    <svg viewBox="0 0 480 300" fill="none" xmlns="http://www.w3.org/2000/svg"
      style={{ width: '100%', height: 'auto', borderRadius: 12 }}>
      <rect width="480" height="300" rx="12" fill="#0D1B2A"/>
      {/* Grid lines */}
      {[60,120,180,240,300,360,420].map(x => (
        <line key={x} x1={x} y1="0" x2={x} y2="300" stroke="rgba(255,255,255,0.04)" strokeWidth="1"/>
      ))}
      {[60,120,180,240].map(y => (
        <line key={y} x1="0" y1={y} x2="480" y2={y} stroke="rgba(255,255,255,0.04)" strokeWidth="1"/>
      ))}

      {/* Left panel: gauge */}
      <rect x="16" y="16" width="210" height="200" rx="8" fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.08)" strokeWidth="0.5"/>
      <text x="30" y="38" fontFamily="monospace" fontSize="8" fill="rgba(196,98,45,0.85)" letterSpacing="1.5">DEFAULT RISK SCORE</text>

      {/* Gauge track */}
      <path d="M 50 170 A 71 71 0 0 1 192 170" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="9" strokeLinecap="round"/>
      {/* Gauge fill — 15% */}
      <path d="M 50 170 A 71 71 0 0 1 86 103" fill="none" stroke="#4A7C59" strokeWidth="9" strokeLinecap="round"/>
      {/* Needle */}
      <line x1="121" y1="170" x2="89" y2="107" stroke="#C4622D" strokeWidth="2" strokeLinecap="round"/>
      <circle cx="121" cy="170" r="5" fill="#C4622D"/>

      <text x="121" y="148" textAnchor="middle" fontFamily="monospace" fontSize="22" fill="white" fontWeight="700">15%</text>
      <text x="121" y="164" textAnchor="middle" fontFamily="monospace" fontSize="8" fill="#4A7C59" letterSpacing="2">● LOW RISK</text>

      {/* Scale labels */}
      <text x="46" y="183" fontFamily="monospace" fontSize="7" fill="rgba(255,255,255,0.25)">0%</text>
      <text x="188" y="183" fontFamily="monospace" fontSize="7" fill="rgba(255,255,255,0.25)">100%</text>

      {/* Risk legend */}
      <rect x="30" y="195" width="8" height="7" rx="2" fill="#4A7C59"/>
      <text x="42" y="201" fontFamily="monospace" fontSize="6.5" fill="rgba(255,255,255,0.35)">Low</text>
      <rect x="78" y="195" width="8" height="7" rx="2" fill="#D4A017"/>
      <text x="90" y="201" fontFamily="monospace" fontSize="6.5" fill="rgba(255,255,255,0.35)">Medium</text>
      <rect x="138" y="195" width="8" height="7" rx="2" fill="#C4622D"/>
      <text x="150" y="201" fontFamily="monospace" fontSize="6.5" fill="rgba(255,255,255,0.35)">High</text>

      {/* Right panel: credit limit */}
      <rect x="242" y="16" width="222" height="200" rx="8" fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.08)" strokeWidth="0.5"/>
      <text x="258" y="38" fontFamily="monospace" fontSize="8" fill="rgba(196,98,45,0.85)" letterSpacing="1.5">CREDIT LIMIT REC.</text>

      <text x="258" y="68" fontFamily="monospace" fontSize="9" fill="rgba(255,255,255,0.35)">Recommended</text>
      <text x="258" y="96" fontFamily="monospace" fontSize="28" fill="white" fontWeight="700">₦150K</text>
      <text x="258" y="112" fontFamily="monospace" fontSize="9" fill="#4A7C59">↑ +50% from current</text>

      <line x1="258" y1="126" x2="448" y2="126" stroke="rgba(255,255,255,0.06)" strokeWidth="1"/>
      <text x="258" y="146" fontFamily="monospace" fontSize="8" fill="rgba(255,255,255,0.3)">Adjustment Factor</text>
      <text x="258" y="172" fontFamily="monospace" fontSize="28" fill="#C4622D" fontWeight="700">×1.5</text>

      <line x1="258" y1="183" x2="448" y2="183" stroke="rgba(255,255,255,0.06)" strokeWidth="1"/>
      <circle cx="265" cy="197" r="3" fill="#4A7C59"/>
      <text x="273" y="201" fontFamily="monospace" fontSize="7.5" fill="rgba(255,255,255,0.5)">Good payment history</text>
      <circle cx="265" cy="210" r="3" fill="#4A7C59"/>
      <text x="273" y="214" fontFamily="monospace" fontSize="7.5" fill="rgba(255,255,255,0.5)">High pay/bill ratio</text>

      {/* Bottom params strip */}
      <rect x="16" y="228" width="448" height="56" rx="7" fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.05)" strokeWidth="0.5"/>
      <text x="30" y="246" fontFamily="monospace" fontSize="7" fill="rgba(196,98,45,0.6)" letterSpacing="1.5">INPUT PARAMETERS</text>
      <text x="30" y="264" fontFamily="monospace" fontSize="7.5" fill="rgba(255,255,255,0.4)">Limit: 100,000</text>
      <text x="120" y="264" fontFamily="monospace" fontSize="7.5" fill="rgba(255,255,255,0.4)">Age: 34</text>
      <text x="190" y="264" fontFamily="monospace" fontSize="7.5" fill="rgba(255,255,255,0.4)">Status: Paid</text>
      <text x="290" y="264" fontFamily="monospace" fontSize="7.5" fill="rgba(255,255,255,0.4)">Bill: 45,200</text>
      <text x="380" y="264" fontFamily="monospace" fontSize="7.5" fill="rgba(255,255,255,0.4)">Paid: 10,000</text>
    </svg>
  );
}

export function LandingPage({ onStart }: { onStart: () => void }) {
  const [activeInfo, setActiveInfo] = React.useState<string | null>(null);

  const infoContent: Record<string, { title: string; text: string }> = {
    platform: {
      title: 'Enterprise Platform',
      text: 'Cloud-native infrastructure built for high-throughput institutional lending, featuring real-time data sync and SOC2-compliant security.',
    },
    'risk-engine': {
      title: 'Neural Risk Engine',
      text: 'TensorFlow deep networks with dropout regularization and random oversampling for robust credit default classification and limit regression.',
    },
    'case-studies': {
      title: 'Institutional Success',
      text: 'See how Tier-1 lenders reduced NPL ratios by 22% within the first two quarters of CreditLens integration.',
    },
    pricing: {
      title: 'Tiered Licensing',
      text: 'Per-call rates for startups to flat enterprise licenses for global deployments. MIT open-source core, premium support tiers.',
    },
  };

  const navItems = [
    { key: 'platform', label: 'Platform' },
    { key: 'risk-engine', label: 'Risk Engine' },
    { key: 'case-studies', label: 'Case Studies' },
    { key: 'pricing', label: 'Pricing' },
  ];

  return (
    <div className="min-h-screen bg-mist">

      {/* ── NAV ─────────────────────────────────────────────────── */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/95 backdrop-blur-md border-b border-rule">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-md bg-gradient-to-br from-copper to-slate-blue flex items-center justify-center">
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                <path d="M3 12L8 4L13 12H3Z" fill="white" fillOpacity="0.9"/>
              </svg>
            </div>
            <span className="font-heading text-lg text-ink">CreditLens</span>
          </div>

          {/* Nav links — position:relative so popover anchors correctly */}
          <div className="relative flex items-center gap-6">
            {navItems.map(item => (
              <button
                key={item.key}
                onClick={() => setActiveInfo(activeInfo === item.key ? null : item.key)}
                className={`font-body text-sm transition-colors cursor-pointer ${
                  activeInfo === item.key
                    ? 'text-copper font-bold border-b-2 border-copper pb-0.5'
                    : 'text-slate-600 hover:text-copper'
                }`}
              >
                {item.label}
              </button>
            ))}

            {/* Popover — anchored to the nav-links wrapper */}
            <AnimatePresence>
              {activeInfo && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 8 }}
                  className="absolute top-full right-0 mt-4 w-64 bg-white border border-rule shadow-lifted rounded-xl p-5 z-50"
                >
                  <div className="flex justify-between items-start mb-3">
                    <h4 className="font-heading text-base text-ink leading-tight">
                      {infoContent[activeInfo].title}
                    </h4>
                    <button onClick={() => setActiveInfo(null)} className="text-slate-400 hover:text-ink ml-2 mt-0.5">
                      <X size={14} />
                    </button>
                  </div>
                  <p className="font-body text-xs text-slate-500 leading-relaxed">
                    {infoContent[activeInfo].text}
                  </p>
                  <div className="mt-4 pt-4 border-t border-rule">
                    <button className="font-mono text-[10px] text-copper font-bold uppercase tracking-widest flex items-center gap-1.5 hover:gap-2.5 transition-all">
                      Learn More <ExternalLink size={11} />
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* CTA */}
          <div className="flex items-center gap-4">
            <button className="font-body text-sm text-slate-600 hover:text-copper transition-colors">Log In</button>
            <button
              onClick={onStart}
              className="bg-copper text-white font-mono text-xs font-bold uppercase tracking-widest px-5 py-2.5 rounded-lg hover:bg-copper-dk transition-colors shadow-ambient active:scale-95"
            >
              Get Started
            </button>
          </div>
        </div>
      </nav>

      {/* ── HERO ────────────────────────────────────────────────── */}
      <header className="pt-32 pb-20 px-6">
        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          <motion.div initial={{ opacity: 0, x: -24 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.6 }}>
            <span className="font-mono text-[10px] text-copper uppercase tracking-widest bg-copper/10 border border-copper/25 px-3 py-1.5 rounded inline-block mb-6">
              Institutional Credit Intelligence
            </span>
            <h1 className="font-heading text-5xl lg:text-6xl text-ink leading-[1.08] mb-6 tracking-tight">
              Know Before<br/>
              <span className="text-copper">You Lend.</span>
            </h1>
            <p className="font-body text-lg text-slate-600 leading-relaxed mb-8 max-w-md">
              Neural network–driven credit risk scoring and limit recommendation. Built for financial institutions that act on data, not guesswork.
            </p>
            <div className="flex gap-4 flex-wrap">
              <button onClick={onStart}
                className="font-mono text-xs font-bold uppercase tracking-widest bg-copper text-white px-7 py-3.5 rounded-lg hover:bg-copper-dk transition-colors shadow-ambient flex items-center gap-2 active:scale-95">
                Try the Tool <ArrowRight size={16} />
              </button>
              <button className="font-mono text-xs font-bold uppercase tracking-widest border-[1.5px] border-slate-blue text-slate-blue px-7 py-3.5 rounded-lg hover:bg-mist transition-colors">
                View API Docs
              </button>
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.6, delay: 0.15 }}>
            {/* Inline SVG — no CDN dependency */}
            <div className="relative">
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-copper/10 to-slate-blue/10 -m-3 blur-2xl" />
              <div className="relative rounded-2xl overflow-hidden border border-rule shadow-lifted">
                <HeroIllustration />
              </div>
              {/* Floating pill */}
              <div className="absolute -bottom-4 -left-4 bg-ink text-white px-5 py-3 rounded-xl shadow-lifted hidden lg:block">
                <p className="font-mono text-[9px] text-white/50 uppercase tracking-widest mb-1">Prob. of Default</p>
                <p className="font-mono text-xl text-amber font-bold">0.150</p>
                <div className="flex items-center gap-1.5 mt-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-sage"></span>
                  <span className="font-mono text-[9px] text-white/70 uppercase tracking-wider">Low Risk Grade</span>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </header>

      {/* ── STATS STRIP ─────────────────────────────────────────── */}
      <div className="bg-white border-y border-rule py-8 px-6">
        <div className="max-w-6xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8">
          {[
            { val: '98.2%', label: 'Model Accuracy' },
            { val: '<200ms', label: 'API Latency' },
            { val: 'MIT', label: 'Open Source' },
            { val: 'TF/Keras', label: 'Neural Networks' },
          ].map(({ val, label }) => (
            <div key={label} className="text-center">
              <div className="font-mono text-2xl font-bold text-copper mb-1">{val}</div>
              <div className="font-mono text-[10px] text-slate-500 uppercase tracking-widest">{label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── HOW IT WORKS ────────────────────────────────────────── */}
      <section className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <span className="font-mono text-[10px] text-copper uppercase tracking-widest">How It Works</span>
            <h2 className="font-heading text-4xl text-ink mt-3">From data to decision in three steps.</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12 relative">
            <div className="hidden md:block absolute top-8 left-[22%] right-[22%] h-px bg-gradient-to-r from-rule via-copper/40 to-rule" />
            {[
              { n: '01', title: 'Submit Customer Data', body: 'Securely ingest financial data via the dashboard form or batch API endpoint.', icon: Database },
              { n: '02', title: 'ML Engine Scores Risk', body: 'Deep networks with dropout regularisation assess default probability and optimal limits.', icon: Zap },
              { n: '03', title: 'Receive Actionable Insight', body: 'Get a risk %, tier classification, and explainable factors for your underwriting workflow.', icon: BarChart3 },
            ].map(({ n, title, body, icon: Icon }, i) => (
              <div key={n} className="text-center relative z-10">
                <div className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6 font-mono text-base font-bold ${
                  i === 1
                    ? 'bg-copper text-white'
                    : 'bg-white border-2 border-rule text-copper shadow-ambient'
                }`}>
                  {n}
                </div>
                <h3 className="font-heading text-lg text-ink mb-3">{title}</h3>
                <p className="font-body text-sm text-slate-500 leading-relaxed">{body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── FEATURES ────────────────────────────────────────────── */}
      <section className="py-24 px-6 bg-white border-y border-rule">
        <div className="max-w-6xl mx-auto">
          <span className="font-mono text-[10px] text-copper uppercase tracking-widest">Capabilities</span>
          <h2 className="font-heading text-4xl text-ink mt-3 mb-12 max-w-lg">
            Everything your credit team needs to act with confidence.
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {[
              { title: 'Default Prediction', desc: 'Binary classification with probability output. Handles imbalanced datasets via random oversampling.', icon: Search, accent: '#C4622D' },
              { title: 'Credit Limit Recommendation', desc: 'Regression engine estimates appropriate limits from payment history, age, education, and bill ratios.', icon: TrendingUp, accent: '#2E4057' },
              { title: 'FastAPI Backend', desc: 'Async REST API with Pydantic validation. Clean JSON in, clean JSON out — integrates in minutes.', icon: Zap, accent: '#4A7C59' },
              { title: 'Transparent Risk Factors', desc: 'Every prediction ships with human-readable factors — the reasoning behind it for your compliance team.', icon: Shield, accent: '#D4A017' },
            ].map(({ title, desc, icon: Icon, accent }) => (
              <div key={title}
                className="bg-mist border border-rule rounded-xl p-7 card-hover cursor-default"
                style={{ borderLeft: `3px solid ${accent}` }}>
                <Icon size={22} style={{ color: accent }} className="mb-4" />
                <h3 className="font-heading text-lg text-ink mb-2">{title}</h3>
                <p className="font-body text-sm text-slate-500 leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── API PREVIEW ──────────────────────────────────────────── */}
      <section className="py-24 px-6 bg-ink">
        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-16 items-start">
          <div>
            <span className="font-mono text-[10px] text-copper uppercase tracking-widest">API Reference</span>
            <h2 className="font-heading text-4xl text-white mt-3 mb-5 leading-tight">Simple endpoints.<br/>Powerful predictions.</h2>
            <p className="font-body text-white/60 leading-relaxed mb-8">
              Three endpoints. Clean JSON in and out. Integrate with any language in minutes.
            </p>
            {[
              { method: 'GET', path: '/api/v1/health', desc: 'Service health & model status' },
              { method: 'POST', path: '/api/v1/predict/default', desc: 'Default risk probability' },
              { method: 'POST', path: '/api/v1/predict/credit-limit', desc: 'Credit limit recommendation' },
            ].map(({ method, path, desc }) => (
              <div key={path} className="flex items-start gap-4 py-4 border-b border-white/06">
                <span className={`font-mono text-[9px] font-bold px-2 py-1 rounded uppercase tracking-widest flex-shrink-0 mt-1 ${
                  method === 'GET' ? 'bg-sage/20 text-sage' : 'bg-copper/20 text-copper'
                }`}>{method}</span>
                <div>
                  <div className="font-mono text-sm text-white/85">{path}</div>
                  <div className="font-body text-xs text-white/40 mt-0.5">{desc}</div>
                </div>
              </div>
            ))}
            <ul className="mt-8 space-y-3">
              {['Standardized OpenAPI 3.0 spec', 'Sandbox environment available', 'Python & TypeScript clients'].map(item => (
                <li key={item} className="flex items-center gap-3 font-body text-sm text-white/70">
                  <CheckCircle2 className="text-sage flex-shrink-0" size={16} /> {item}
                </li>
              ))}
            </ul>
          </div>

          <div className="bg-[#0A1520] border border-white/08 rounded-xl overflow-hidden">
            <div className="flex items-center gap-2 px-5 py-3 bg-black/30 border-b border-white/06">
              {['#F7685B','#FDBC40','#34C84A'].map(c => (
                <div key={c} style={{ width: 10, height: 10, borderRadius: '50%', background: c }}/>
              ))}
              <span className="font-mono text-[10px] text-white/30 ml-2">predict/default · curl</span>
            </div>
            <pre className="font-mono text-xs text-white/70 leading-relaxed p-5 overflow-x-auto">{
`curl -X POST \\
  http://localhost:8000/api/v1/predict/default \\
  -H "Content-Type: application/json" \\
  -d '{
    "credit_limit": 100000,
    "age": 34,
    "payment_status": 0,
    "bill_amount": 45200,
    "payment_amount": 10000
  }'`}</pre>
            <div className="border-t border-white/06 px-5 pt-3 pb-5">
              <div className="font-mono text-[9px] text-sage uppercase tracking-widest mb-2">Response</div>
              <pre className="font-mono text-xs text-white/50 leading-relaxed">{
`{
  "prediction": "Low Risk of Default",
  "probability": 0.15,
  "is_high_risk": false,
  "risk_factors": [
    "Low payment to bill ratio"
  ]
}`}</pre>
            </div>
          </div>
        </div>
      </section>

      {/* ── CTA ─────────────────────────────────────────────────── */}
      <section className="py-24 px-6 bg-copper relative overflow-hidden text-center">
        <div className="absolute inset-0 opacity-5"
          style={{ backgroundImage: 'repeating-linear-gradient(45deg, white 0, white 1px, transparent 1px, transparent 20px)' }}/>
        <div className="relative max-w-2xl mx-auto">
          <h2 className="font-heading text-5xl text-white mb-5 tracking-tight">
            Start predicting smarter credit decisions.
          </h2>
          <p className="font-body text-white/75 text-lg mb-8">
            Open source, MIT licensed. Clone and run your first prediction in under five minutes.
          </p>
          <div className="flex gap-4 justify-center flex-wrap">
            <a href="https://github.com/austinLorenzMccoy/credit-default-prediction"
              className="font-mono text-xs font-bold uppercase tracking-widest bg-white text-copper px-8 py-3.5 rounded-lg hover:opacity-90 transition-opacity">
              View on GitHub →
            </a>
            <button onClick={onStart}
              className="font-mono text-xs font-bold uppercase tracking-widest border border-white/40 text-white px-8 py-3.5 rounded-lg hover:border-white transition-colors">
              Open Dashboard
            </button>
          </div>
        </div>
      </section>

      {/* ── FOOTER ──────────────────────────────────────────────── */}
      <footer className="bg-ink py-10 px-6 border-t border-white/06">
        <div className="max-w-6xl mx-auto flex items-center justify-between flex-wrap gap-6">
          <div className="flex items-center gap-2.5">
            <div className="w-6 h-6 rounded bg-gradient-to-br from-copper to-slate-blue flex items-center justify-center">
              <svg width="10" height="10" viewBox="0 0 16 16" fill="none">
                <path d="M3 12L8 4L13 12H3Z" fill="white" fillOpacity="0.9"/>
              </svg>
            </div>
            <span className="font-heading text-sm text-white">CreditLens</span>
            <span className="font-mono text-[9px] text-white/25 uppercase tracking-widest ml-1">MIT License</span>
          </div>
          <div className="flex gap-6">
            {[
              { label: 'GitHub', href: 'https://github.com/austinLorenzMccoy/credit-default-prediction' },
              { label: 'chibuezeaugustine23@gmail.com', href: 'mailto:chibuezeaugustine23@gmail.com' },
            ].map(({ label, href }) => (
              <a key={label} href={href} className="font-mono text-[11px] text-white/35 hover:text-white/70 transition-colors tracking-wide">
                {label}
              </a>
            ))}
          </div>
        </div>
      </footer>
    </div>
  );
}
