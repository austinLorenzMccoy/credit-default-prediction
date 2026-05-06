"use client"

import Link from "next/link"
import { ArrowRight, FileText, Brain, Zap, Eye, ChevronRight, Upload, Cpu, BarChart3, Menu, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useState } from "react"

export default function LandingPage() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  
  return (
    <div className="min-h-screen bg-ink">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-ink/95 backdrop-blur-sm border-b border-white/5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-md bg-gradient-to-br from-copper to-slate-blue flex items-center justify-center">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M3 12L8 4L13 12H3Z" fill="white" fillOpacity="0.9"/>
                <circle cx="8" cy="8" r="2" fill="white" fillOpacity="0.6"/>
              </svg>
            </div>
            <span className="font-serif text-lg font-bold text-white">CreditLens</span>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden lg:flex items-center gap-8">
            <div className="flex items-center gap-6">
              {["Solutions", "Platform", "Risk Engine", "Case Studies", "Pricing"].map((item, i) => (
                <Link 
                  key={item} 
                  href="#" 
                  className={`font-mono text-sm transition-colors ${i === 0 ? "text-copper" : "text-white/60 hover:text-sand"}`}
                >
                  {item}
                </Link>
              ))}
            </div>
            <div className="flex items-center gap-4">
              <Link href="#" className="font-mono text-sm text-white/60 hover:text-sand transition-colors">
                Log In
              </Link>
              <Link href="/dashboard">
                <Button className="bg-copper hover:bg-copper-dark text-white font-mono text-sm px-4 py-2">
                  Get Started
                </Button>
              </Link>
            </div>
          </div>

          {/* Tablet: Show Log In + Get Started */}
          <div className="hidden md:flex lg:hidden items-center gap-3">
            <Link href="#" className="font-mono text-sm text-white/60 hover:text-sand transition-colors">
              Log In
            </Link>
            <Link href="/dashboard">
              <Button className="bg-copper hover:bg-copper-dark text-white font-mono text-sm px-4 py-2">
                Get Started
              </Button>
            </Link>
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="ml-2 p-2 text-white/60 hover:text-white transition-colors"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>

          {/* Mobile: Show Get Started + Menu */}
          <div className="flex md:hidden items-center gap-2">
            <Link href="/dashboard">
              <Button className="bg-copper hover:bg-copper-dark text-white font-mono text-xs px-3 py-1.5">
                Get Started
              </Button>
            </Link>
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 text-white/60 hover:text-white transition-colors"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>
        </div>

        {/* Mobile Menu Dropdown */}
        {mobileMenuOpen && (
          <div className="lg:hidden bg-ink/98 border-t border-white/5 backdrop-blur-sm">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4 space-y-1">
              {["Solutions", "Platform", "Risk Engine", "Case Studies", "Pricing"].map((item, i) => (
                <Link 
                  key={item} 
                  href="#" 
                  className={`block py-2 font-mono text-sm transition-colors ${i === 0 ? "text-copper" : "text-white/60 hover:text-sand"}`}
                  onClick={() => setMobileMenuOpen(false)}
                >
                  {item}
                </Link>
              ))}
              <div className="pt-3 mt-3 border-t border-white/10">
                <Link 
                  href="#" 
                  className="block py-2 font-mono text-sm text-white/60 hover:text-sand transition-colors md:hidden"
                  onClick={() => setMobileMenuOpen(false)}
                >
                  Log In
                </Link>
              </div>
            </div>
          </div>
        )}
      </nav>

      {/* Hero Section */}
      <section className="relative pt-32 pb-24 overflow-hidden">
        {/* Background elements */}
        <div className="absolute inset-0 bg-gradient-to-br from-ink via-ink to-slate-blue/30" />
        <div className="absolute right-0 top-1/4 w-[500px] h-[500px] bg-copper/10 rounded-full blur-[120px]" />
        
        <div className="relative max-w-7xl mx-auto px-6">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            {/* Left Column - Text */}
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-copper/10 border border-copper/30 rounded-full mb-6">
                <span className="font-mono text-[11px] text-copper tracking-wider uppercase">Institutional Credit Intelligence</span>
              </div>
              
              <h1 className="font-serif text-5xl lg:text-6xl font-bold text-white leading-[1.1] mb-6">
                <span className="italic">Know Before You</span><br />
                <span className="text-copper">Lend.</span>
              </h1>
              
              <p className="font-sans text-lg text-white/60 leading-relaxed mb-8 max-w-md">
                Neural network–driven credit risk scoring and limit recommendation. Built for financial institutions that act on data, not guesswork.
              </p>
              
              <div className="flex flex-wrap items-center gap-4">
                <Link href="/dashboard">
                  <Button className="bg-copper hover:bg-copper-dark text-white font-mono px-6 py-3 h-auto">
                    Try the Tool
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
                <Link href="#">
                  <Button variant="outline" className="border-white/20 text-sand hover:bg-white/5 font-mono px-6 py-3 h-auto">
                    View API Docs
                  </Button>
                </Link>
              </div>
            </div>

            {/* Right Column - Dashboard Preview */}
            <div className="relative">
              <div className="bg-white/5 border border-white/10 rounded-2xl p-6 backdrop-blur-sm">
                <div className="absolute -top-px left-8 right-8 h-0.5 bg-gradient-to-r from-transparent via-copper to-transparent" />
                
                {/* Mock Dashboard */}
                <div className="bg-gradient-to-br from-slate-blue/50 to-ink rounded-xl p-6 mb-4 border border-white/5">
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    {/* Risk Gauge */}
                    <div className="bg-white/5 rounded-lg p-4">
                      <div className="font-mono text-[10px] text-copper uppercase tracking-wider mb-3">Default Risk</div>
                      <div className="flex items-center justify-center">
                        <svg viewBox="0 0 100 60" className="w-full max-w-[120px]">
                          <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="8" strokeLinecap="round"/>
                          <path d="M 10 50 A 40 40 0 0 1 40 15" fill="none" stroke="#4A7C59" strokeWidth="8" strokeLinecap="round"/>
                          <text x="50" y="45" textAnchor="middle" className="font-mono text-sm font-bold fill-white">15%</text>
                        </svg>
                      </div>
                      <div className="text-center mt-2">
                        <span className="inline-block px-2 py-0.5 bg-sage/20 text-sage font-mono text-[10px] rounded">LOW RISK</span>
                      </div>
                    </div>

                    {/* Credit Limit */}
                    <div className="bg-white/5 rounded-lg p-4">
                      <div className="font-mono text-[10px] text-copper uppercase tracking-wider mb-3">Credit Limit</div>
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-2 bg-slate-blue rounded-full" />
                          <span className="font-mono text-xs text-white/60">100K</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-2 bg-copper rounded-full" />
                          <span className="font-mono text-xs text-copper">150K</span>
                        </div>
                      </div>
                      <div className="mt-3">
                        <span className="font-mono text-lg font-bold text-copper">×1.5</span>
                      </div>
                    </div>
                  </div>

                  {/* Factors Strip */}
                  <div className="bg-white/5 rounded-lg p-3">
                    <div className="font-mono text-[9px] text-copper/70 uppercase tracking-wider mb-2">Recommendation Factors</div>
                    <div className="flex flex-wrap gap-2">
                      {["Good payment history", "High pay/bill ratio", "Education positive"].map(factor => (
                        <div key={factor} className="flex items-center gap-1.5">
                          <div className="h-1.5 w-1.5 rounded-full bg-sage" />
                          <span className="font-mono text-[10px] text-white/70">{factor}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Stats Row */}
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: "Default Risk", value: "15.00%", tag: "Low Risk", tagColor: "sage" },
                    { label: "Credit Limit", value: "₦150,000", tag: "+50% increase", tagColor: "copper" },
                  ].map(({ label, value, tag, tagColor }) => (
                    <div key={label} className="bg-white/5 border border-white/10 rounded-lg p-4">
                      <div className="font-mono text-[10px] text-white/40 uppercase tracking-wider mb-1">{label}</div>
                      <div className="font-mono text-xl font-semibold text-white mb-2">{value}</div>
                      <span className={`inline-block px-2 py-0.5 rounded font-mono text-[10px] ${
                        tagColor === "sage" ? "bg-sage/20 text-sage border border-sage/30" : "bg-copper/20 text-copper border border-copper/30"
                      }`}>{tag}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Floating Badge */}
              <div className="absolute -bottom-4 left-1/2 -translate-x-1/2 bg-ink border border-copper/40 rounded-lg px-4 py-3 shadow-xl">
                <div className="flex items-center gap-3">
                  <div className="font-mono text-[10px] text-white/50 uppercase">Probability of Default</div>
                  <div className="font-mono text-2xl font-bold text-copper">0.024</div>
                  <div className="px-2 py-0.5 bg-sage/20 text-sage font-mono text-[10px] rounded">LOW RISK GRADE</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Strip */}
      <section className="py-8 bg-sand border-y border-rule">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center justify-center gap-16">
            {[
              { value: "98.2%", label: "Accuracy" },
              { value: "<200ms", label: "Latency" },
              { value: "MIT", label: "Open Source" },
              { value: "Neural", label: "ML Engine" },
            ].map(({ value, label }) => (
              <div key={label} className="text-center">
                <div className="font-mono text-2xl font-bold text-copper">{value}</div>
                <div className="font-mono text-xs text-muted-foreground uppercase tracking-wider">{label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-24 bg-mist">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <span className="font-mono text-xs text-copper uppercase tracking-wider">Precision Workflow</span>
            <h2 className="font-serif text-3xl font-bold text-ink mt-3">
              From raw data to board-ready insights in milliseconds.
            </h2>
            <p className="font-sans text-muted-foreground mt-2">Our pipeline is designed for security and speed.</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 relative">
            {/* Connector Line */}
            <div className="hidden md:block absolute top-12 left-[16%] right-[16%] h-px bg-gradient-to-r from-rule via-copper/40 to-rule" />

            {[
              { icon: Upload, title: "Submit Customer Data", desc: "Securely ingest financial statements and behavioral data via our robust API or batch CSV processing." },
              { icon: Cpu, title: "ML Engine Scores Risk", desc: "Deep learning models process hundreds of variables to identify non-linear risk patterns invisible to legacy systems." },
              { icon: BarChart3, title: "Receive Actionable Insight", desc: "Get clear, documented recommendations including probability of default and recommended exposure limits." },
            ].map(({ icon: Icon, title, desc }, i) => (
              <div key={title} className="text-center relative">
                <div className={`mx-auto w-20 h-20 rounded-full flex items-center justify-center mb-6 relative z-10 ${
                  i === 1 ? "bg-copper text-white" : "bg-white text-copper border-2 border-rule"
                }`}>
                  <Icon className="h-8 w-8" />
                </div>
                <h3 className="font-serif text-xl text-ink mb-3">{title}</h3>
                <p className="font-sans text-sm text-muted-foreground leading-relaxed max-w-xs mx-auto">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-24 bg-white border-t border-rule">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { icon: FileText, title: "Default Prediction", desc: "Probability of default modeling using transformer-based architectures for higher accuracy.", color: "copper" },
              { icon: Brain, title: "Limit Recommendation", desc: "Dynamic credit line suggestions optimized for portfolio-wide risk distribution and liquidity.", color: "slate-blue" },
              { icon: Zap, title: "FastAPI Backend", desc: "Built for high-concurrency requests with asynchronous processing for real-time risk evaluation.", color: "sage" },
              { icon: Eye, title: "Transparent Factors", desc: "Explainable AI (XAI) layers that highlight the specific variables driving each risk score.", color: "amber" },
            ].map(({ icon: Icon, title, desc, color }) => (
              <div key={title} className={`bg-mist border border-rule rounded-xl p-6 border-l-4 border-l-${color} hover:shadow-lg transition-shadow`}
                style={{ borderLeftColor: color === "copper" ? "#C4622D" : color === "slate-blue" ? "#2E4057" : color === "sage" ? "#4A7C59" : "#D4A017" }}
              >
                <Icon className={`h-8 w-8 mb-4`} style={{ color: color === "copper" ? "#C4622D" : color === "slate-blue" ? "#2E4057" : color === "sage" ? "#4A7C59" : "#D4A017" }} />
                <h3 className="font-serif text-lg text-ink mb-2">{title}</h3>
                <p className="font-sans text-sm text-muted-foreground leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* API Preview */}
      <section className="py-24 bg-ink border-t border-white/5">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid lg:grid-cols-2 gap-16 items-start">
            {/* Left - Text */}
            <div>
              <span className="font-mono text-xs text-copper uppercase tracking-wider">Developer First</span>
              <h2 className="font-serif text-3xl font-bold text-white mt-3 mb-4">Integrate in Minutes</h2>
              <p className="font-sans text-white/60 leading-relaxed mb-8">
                Our API is designed for financial engineers. Clean endpoints, comprehensive documentation, 
                and predictable JSON responses make integration into your existing LOS simple.
              </p>

              <div className="space-y-4">
                {[
                  "Standardized OpenAPI 3.0 specification",
                  "Sandbox environment for stress testing",
                  "Python, Node, and Go SDKs available",
                ].map(item => (
                  <div key={item} className="flex items-center gap-3">
                    <div className="h-5 w-5 rounded-full bg-sage/20 flex items-center justify-center">
                      <svg className="h-3 w-3 text-sage" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                    <span className="font-sans text-white/70">{item}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Right - Code Preview */}
            <div className="bg-[#0A1520] border border-white/10 rounded-xl overflow-hidden">
              {/* Tab Bar */}
              <div className="flex items-center bg-black/30 border-b border-white/5 px-4">
                <div className="flex gap-1.5 py-3 pr-6">
                  <div className="w-3 h-3 rounded-full bg-[#F7685B]" />
                  <div className="w-3 h-3 rounded-full bg-[#FDBC40]" />
                  <div className="w-3 h-3 rounded-full bg-[#34C84A]" />
                </div>
                <div className="font-mono text-xs text-copper border-b-2 border-copper px-3 py-3">Request</div>
              </div>

              <div className="p-6 font-mono text-xs leading-relaxed">
                <div className="text-white/50">POST /v1/predict/default</div>
                <div className="mt-4 text-white/70">
                  {`{`}<br/>
                  &nbsp;&nbsp;<span className="text-copper">&quot;entity_id&quot;</span>: <span className="text-sage">&quot;FIN-8821&quot;</span>,<br/>
                  &nbsp;&nbsp;<span className="text-copper">&quot;metrics&quot;</span>: {`{`}<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;<span className="text-copper">&quot;cash_ratio&quot;</span>: <span className="text-amber">0.82</span>,<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;<span className="text-copper">&quot;debt_to_equity&quot;</span>: <span className="text-amber">1.45</span>,<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;<span className="text-copper">&quot;monthly_revenue&quot;</span>: <span className="text-amber">450000</span><br/>
                  &nbsp;&nbsp;{`}`}<br/>
                  {`}`}
                </div>

                <div className="mt-6 pt-4 border-t border-white/5">
                  <div className="text-white/50 mb-3">// Response</div>
                  <div className="text-white/70">
                    {`{`}<br/>
                    &nbsp;&nbsp;<span className="text-copper">&quot;risk_score&quot;</span>: <span className="text-amber">0.14</span>,<br/>
                    &nbsp;&nbsp;<span className="text-copper">&quot;grade&quot;</span>: <span className="text-sage">&quot;A-&quot;</span>,<br/>
                    &nbsp;&nbsp;<span className="text-copper">&quot;recommendation&quot;</span>: <span className="text-sage">&quot;APPROVE&quot;</span>,<br/>
                    &nbsp;&nbsp;<span className="text-copper">&quot;max_limit&quot;</span>: <span className="text-amber">250000</span><br/>
                    {`}`}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-24 bg-gradient-to-br from-copper to-copper-dark relative overflow-hidden">
        <div className="absolute inset-0 bg-[repeating-linear-gradient(45deg,rgba(255,255,255,0.03)_0px,rgba(255,255,255,0.03)_1px,transparent_1px,transparent_20px)]" />
        <div className="relative max-w-4xl mx-auto px-6 text-center">
          <h2 className="font-serif text-4xl font-bold text-white mb-4">
            Start predicting smarter credit decisions.
          </h2>
          <p className="font-sans text-white/80 mb-8 max-w-lg mx-auto">
            Open source, MIT licensed, and ready to deploy. Clone the repo and run your first prediction in under five minutes.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-4">
            <Link href="https://github.com" target="_blank">
              <Button className="bg-white text-copper hover:bg-white/90 font-mono px-6 py-3 h-auto">
                View on GitHub
                <ChevronRight className="ml-1 h-4 w-4" />
              </Button>
            </Link>
            <Link href="#">
              <Button variant="outline" className="border-white/40 text-white hover:bg-white/10 font-mono px-6 py-3 h-auto">
                Read the Docs
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 bg-ink border-t border-white/5">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-wrap items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              <div className="h-7 w-7 rounded-md bg-gradient-to-br from-copper to-slate-blue flex items-center justify-center">
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                  <path d="M3 12L8 4L13 12H3Z" fill="white" fillOpacity="0.9"/>
                </svg>
              </div>
              <span className="font-serif text-base font-bold text-white">CreditLens</span>
              <span className="font-mono text-[10px] text-white/30 tracking-wider">MIT License</span>
            </div>

            <div className="flex items-center gap-8">
              {["Privacy Policy", "Terms of Service", "Security", "API Status", "Contact Support"].map(item => (
                <Link key={item} href="#" className="font-mono text-xs text-white/40 hover:text-sand transition-colors">
                  {item}
                </Link>
              ))}
            </div>
          </div>
          <div className="mt-8 pt-6 border-t border-white/5 text-center">
            <p className="font-mono text-xs text-white/30">© 2024 CreditLens Intelligence. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
