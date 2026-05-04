import React from 'react';
import { ArrowRight, CheckCircle2, Github, ExternalLink, Zap, Shield, Search, TrendingUp, BarChart3, Database, X } from 'lucide-react';
import { Page } from '../types';
import { motion, AnimatePresence } from 'motion/react';

export function LandingPage({ onStart }: { onStart: () => void }) {
  const [activeInfo, setActiveInfo] = React.useState<string | null>(null);

  const infoContent: Record<string, { title: string, text: string }> = {
    'platform': { 
      title: 'Enterprise Platform', 
      text: 'Our cloud-native infrastructure is built for high-throughput institutional lending, featuring real-time data sync and SOC2-compliant security.' 
    },
    'risk-engine': { 
      title: 'Neural Risk Engine', 
      text: 'Advanced XGBoost and Transformer models trained on billions of transaction records to detect non-linear default patterns.' 
    },
    'case-studies': { 
      title: 'Institutional Success', 
      text: 'See how Tier-1 banks reduced their NPL ratios by 22% within the first two quarters of CreditLens integration.' 
    },
    'pricing': { 
      title: 'Tiered Licensing', 
      text: 'Flexible pricing models ranging from startup-friendly per-call rates to flat enterprise licenses for global deployments.' 
    }
  };

  return (
    <div className="min-h-screen bg-mist">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 border-b border-rule bg-surface/95 backdrop-blur-md shadow-ambient">
        <div className="flex justify-between items-center max-w-7xl mx-auto px-6 h-16">
          <div className="text-xl font-bold text-ink font-heading cursor-pointer" onClick={() => window.scrollTo(0, 0)}>CreditLens</div>
          <div className="hidden md:flex items-center gap-8 relative">
            <button className="text-copper border-b-2 border-copper pb-1 font-serif tracking-tight text-sm cursor-pointer">Solutions</button>
            <div className="relative group">
              <button 
                onClick={() => setActiveInfo(activeInfo === 'platform' ? null : 'platform')}
                className={`font-medium text-sm font-serif tracking-tight hover:text-copper transition-colors cursor-pointer ${activeInfo === 'platform' ? 'text-copper font-bold' : 'text-slate-600'}`}
              >
                Platform
              </button>
            </div>
            <div className="relative group">
              <button 
                onClick={() => setActiveInfo(activeInfo === 'risk-engine' ? null : 'risk-engine')}
                className={`font-medium text-sm font-serif tracking-tight hover:text-copper transition-colors cursor-pointer ${activeInfo === 'risk-engine' ? 'text-copper font-bold' : 'text-slate-600'}`}
              >
                Risk Engine
              </button>
            </div>
            <div className="relative group">
              <button 
                onClick={() => setActiveInfo(activeInfo === 'case-studies' ? null : 'case-studies')}
                className={`font-medium text-sm font-serif tracking-tight hover:text-copper transition-colors cursor-pointer ${activeInfo === 'case-studies' ? 'text-copper font-bold' : 'text-slate-600'}`}
              >
                Case Studies
              </button>
            </div>
            <div className="relative group">
              <button 
                onClick={() => setActiveInfo(activeInfo === 'pricing' ? null : 'pricing')}
                className={`font-medium text-sm font-serif tracking-tight hover:text-copper transition-colors cursor-pointer ${activeInfo === 'pricing' ? 'text-copper font-bold' : 'text-slate-600'}`}
              >
                Pricing
              </button>
            </div>

            {/* Info Popover */}
            <AnimatePresence>
              {activeInfo && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 10 }}
                  className="absolute top-full left-0 mt-4 w-64 bg-white border border-rule shadow-xl rounded-xl p-6 z-50 text-left"
                >
                  <div className="flex justify-between items-center mb-3">
                    <h4 className="font-heading text-lg text-ink leading-none">{infoContent[activeInfo].title}</h4>
                    <button onClick={() => setActiveInfo(null)} className="text-slate-400 hover:text-ink"><X size={14} /></button>
                  </div>
                  <p className="text-xs text-slate-500 font-serif leading-relaxed">
                    {infoContent[activeInfo].text}
                  </p>
                  <div className="mt-4 pt-4 border-t border-rule">
                    <button className="text-[10px] font-mono text-copper font-bold uppercase tracking-widest flex items-center gap-1.5 hover:gap-2 transition-all">
                      Learn More <ExternalLink size={12} />
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          <div className="flex items-center gap-4">
            <button className="text-slate-600 font-medium text-sm font-serif hover:text-copper transition-colors">Log In</button>
            <button 
              onClick={onStart}
              className="bg-copper text-white px-5 py-2 rounded-lg text-sm font-bold shadow-sm hover:opacity-90 active:scale-95 transition-all"
            >
              Get Started
            </button>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <header className="pt-32 pb-20 px-6">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center gap-16">
          <motion.div 
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex-1 text-left"
          >
            <span className="font-mono text-[13px] text-copper uppercase mb-4 block tracking-widest">Institutional Credit Intelligence</span>
            <h1 className="font-display text-display-lg text-ink mb-6">Know Before You Lend.</h1>
            <p className="text-body text-slate-600 max-w-lg mb-10 leading-relaxed text-lg">
              Neural network–driven credit risk scoring and limit recommendation. Built for financial institutions that act on data, not guesswork.
            </p>
            <div className="flex flex-wrap gap-4">
              <button 
                onClick={onStart}
                className="bg-copper text-white px-8 py-3 rounded-lg font-bold flex items-center gap-2 hover:bg-copper/90 transition-all shadow-lg hover:shadow-xl active:scale-95"
              >
                Try the Tool <ArrowRight size={18} />
              </button>
              <button className="border-2 border-copper text-copper px-8 py-3 rounded-lg font-bold hover:bg-copper/5 transition-colors">
                View API Docs
              </button>
            </div>
          </motion.div>
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex-1 w-full relative"
          >
            <div className="relative bg-surface p-4 rounded-xl shadow-ambient border border-rule transform rotate-1">
              <div className="rounded-lg overflow-hidden border border-rule/50">
                <img 
                  src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=800" 
                  alt="Dashboard Preview" 
                  className="w-full h-auto rounded"
                />
              </div>
              {/* Floating Data Pill */}
              <div className="absolute -bottom-6 -left-6 bg-slate-blue text-white p-6 rounded-lg shadow-xl hidden lg:block">
                <div className="font-mono text-[10px] opacity-70 mb-1">PROBABILITY OF DEFAULT</div>
                <div className="font-mono text-2xl text-amber">0.024</div>
                <div className="mt-2 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-sage"></span>
                  <span className="font-mono text-[11px]">LOW RISK GRADE</span>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </header>

      {/* Stats Strip */}
      <section className="bg-mist border-y border-rule py-12">
        <div className="max-w-7xl mx-auto px-6 grid grid-cols-2 md:grid-cols-4 gap-8">
          {[
            { label: 'Accuracy', value: '98.2%' },
            { label: 'Latency', value: '<200ms' },
            { label: 'Open Source', value: 'MIT' },
            { label: 'ML Engine', value: 'Neural' },
          ].map((stat, idx) => (
            <div key={idx} className={`text-center ${idx !== 3 ? 'md:border-r border-rule/60' : ''}`}>
              <div className="font-mono text-copper text-2xl mb-1 font-bold">{stat.value}</div>
              <div className="font-mono text-ink text-sm opacity-70">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* How It Works */}
      <section className="py-24 px-6 bg-surface">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-20">
            <h2 className="font-heading text-4xl text-ink mb-4">Precision Workflow</h2>
            <p className="text-body text-slate-500 max-w-xl mx-auto">From raw data to board-ready insights in milliseconds. Our pipeline is designed for security and speed.</p>
          </div>
          <div className="grid md:grid-cols-3 gap-12 relative">
            <div className="hidden md:block absolute top-[15%] left-[20%] right-[20%] h-px bg-rule/50"></div>
            {[
              { title: 'Submit Customer Data', desc: 'Securely ingest financial statements and behavioral data via our robust API or batch CSV processing.', icon: Database },
              { title: 'ML Engine Scores Risk', desc: 'Deep learning models process hundreds of variables to identify non-linear risk patterns invisible to legacy systems.', icon: Zap },
              { title: 'Receive Actionable Insight', desc: 'Get clear, documented recommendations including probability of default and recommended exposure limits.', icon: BarChart3 },
            ].map((step, idx) => (
              <div key={idx} className="text-center flex flex-col items-center group">
                <div className="w-16 h-16 rounded-full bg-surface-container flex items-center justify-center text-copper mb-6 border border-copper/20 group-hover:scale-110 transition-transform">
                  <step.icon size={28} />
                </div>
                <h3 className="font-heading text-xl text-ink mb-3">{step.title}</h3>
                <p className="text-sm text-slate-500 leading-relaxed px-4">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-24 px-6 bg-mist">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { title: 'Default Prediction', desc: 'Probability of default modeling using transformer-based architectures for higher accuracy.', icon: Search },
              { title: 'Limit Recommendation', desc: 'Dynamic credit line suggestions optimized for portfolio-wide risk distribution and liquidity.', icon: TrendingUp },
              { title: 'FastAPI Backend', desc: 'Built for high-concurrency requests with asynchronous processing for real-time risk evaluation.', icon: Zap },
              { title: 'Transparent Factors', desc: 'Explainable AI (XAI) layers that highlight the specific variables driving each risk score.', icon: Shield },
            ].map((f, idx) => (
              <div key={idx} className="bg-surface p-8 border border-rule shadow-ambient rounded-lg border-t-4 border-t-slate-blue hover:-translate-y-1 transition-all">
                <f.icon className="text-copper mb-6" size={32} />
                <h4 className="font-heading text-lg mb-3">{f.title}</h4>
                <p className="text-sm text-slate-500 leading-relaxed">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* API Preview */}
      <section className="py-24 px-6 bg-surface overflow-hidden">
        <div className="max-w-7xl mx-auto flex flex-col lg:flex-row items-center gap-16">
          <div className="lg:w-1/2">
            <span className="font-mono text-[13px] text-copper uppercase mb-4 block tracking-widest">Developer First</span>
            <h2 className="font-heading text-4xl text-ink mb-6">Integrate in Minutes</h2>
            <p className="text-body text-slate-500 leading-relaxed mb-8 text-lg">
              Our API is designed for financial engineers. Clean endpoints, comprehensive documentation, and predictable JSON responses make integration into your existing LOS simple.
            </p>
            <ul className="space-y-4">
              {['Standardized OpenAPI 3.0 specification', 'Sandbox environment for stress testing', 'Python, Node, and Go SDKs available'].map((item, idx) => (
                <li key={idx} className="flex items-start gap-3">
                  <CheckCircle2 className="text-sage mt-1" size={18} />
                  <span className="text-slate-700">{item}</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="lg:w-1/2 w-full">
            <div className="bg-ink rounded-xl p-6 code-glow relative">
              <div className="flex gap-2 mb-4">
                <div className="w-3 h-3 rounded-full bg-error/40"></div>
                <div className="w-3 h-3 rounded-full bg-amber/40"></div>
                <div className="w-3 h-3 rounded-full bg-sage/40"></div>
              </div>
              <pre className="font-mono text-sm leading-relaxed text-white overflow-x-auto p-2">
                <code className="block">
                  <span className="text-copper">POST</span> /v1/predict/default{'\n'}
                  {'{'}{'\n'}
                  {'  '}<span className="text-amber">"entity_id"</span>: <span className="text-sage">"FIN-8821"</span>,{'\n'}
                  {'  '}<span className="text-amber">"metrics"</span>: {'{'}{'\n'}
                  {'    '}<span className="text-amber">"cash_ratio"</span>: 0.82,{'\n'}
                  {'    '}<span className="text-amber">"debt_to_equity"</span>: 1.45,{'\n'}
                  {'    '}<span className="text-amber">"monthly_revenue"</span>: 450000{'\n'}
                  {'  '}{'}'}{'\n'}
                  {'}'}{'\n\n'}
                  <span className="text-slate-400">// Response</span>{'\n'}
                  {'{'}{'\n'}
                  {'  '}<span className="text-amber">"risk_score"</span>: 0.14,{'\n'}
                  {'  '}<span className="text-amber">"grade"</span>: <span className="text-sage">"A"</span>,{'\n'}
                  {'  '}<span className="text-amber">"recommendation"</span>: <span className="text-sage">"APPROVE"</span>,{'\n'}
                  {'  '}<span className="text-amber">"max_limit"</span>: 250000{'\n'}
                  {'}'}
                </code>
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-surface-container-low py-12 px-6 border-t border-rule mt-20">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="flex flex-col items-center md:items-start gap-2">
            <div className="text-xl font-bold text-ink font-heading">CreditLens</div>
            <p className="text-slate-500 font-serif text-sm">© 2024 CreditLens Intelligence. All rights reserved.</p>
          </div>
          <div className="flex flex-wrap justify-center gap-8 text-sm text-slate-500 font-serif">
            <a href="#" className="hover:text-copper underline decoration-rule transition-colors">Privacy Policy</a>
            <a href="#" className="hover:text-copper underline decoration-rule transition-colors">Terms of Service</a>
            <a href="#" className="hover:text-copper underline decoration-rule transition-colors">Security</a>
            <a href="#" className="hover:text-copper underline decoration-rule transition-colors">API Status</a>
            <a href="#" className="hover:text-copper underline decoration-rule transition-colors">Contact Support</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
