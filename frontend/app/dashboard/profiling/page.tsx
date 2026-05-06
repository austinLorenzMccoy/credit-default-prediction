"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Check, AlertTriangle, Download, GitCompare, RefreshCw, ArrowRight, Lightbulb, Gauge } from "lucide-react"
import Link from "next/link"

export default function ProfilingPage() {
  const limitData = {
    current: 100000,
    recommended: 150000,
    adjustment: 1.5,
    deltaValue: 50000,
    riskExposure: "Low (0.12%)",
    confidence: 94.8,
  }

  const factors = [
    { 
      label: "Cash Flow Consistency", 
      description: "24 months of uninterrupted positive quarterly net growth.",
      status: "positive" 
    },
    { 
      label: "Market Sentiment Index", 
      description: "Top-tier performance in Logistics Sector benchmark.",
      status: "positive" 
    },
    { 
      label: "Account Longevity", 
      description: "Customer tenure exceeds 5-year primary risk threshold.",
      status: "positive" 
    },
    { 
      label: "Macro Environment", 
      description: "Regional inflation rates may impact secondary margins.",
      status: "warning" 
    },
  ]

  const adjustmentFactors = [
    { label: "Revenue Growth Velocity", weight: 40 },
    { label: "Historical Payment Reliability", weight: 25 },
    { label: "Industry Benchmarking (Logistics)", weight: 20 },
    { label: "External Bureau Integration", weight: 15 },
  ]

  return (
    <div className="max-w-7xl mx-auto">
      {/* Breadcrumb and Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 font-mono text-xs text-muted-foreground mb-2">
          <Link href="/dashboard" className="hover:text-copper transition-colors">PROFILING</Link>
          <span className="text-rule">›</span>
          <span className="text-copper">CREDIT LIMIT DETAIL</span>
        </div>

        <div className="flex items-center justify-between">
          <div>
            <h1 className="font-serif text-3xl font-bold text-ink">Limit Recommendation</h1>
            <p className="font-sans text-muted-foreground mt-1">
              Detailed analysis for Global Logistics Corp (GLC-8829)
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Button variant="outline" className="font-mono text-sm border-copper text-copper hover:bg-copper/10">
              VIEW API DOCS
            </Button>
            <Button className="font-mono text-sm bg-copper hover:bg-copper-dark text-white">
              APPROVE INCREASE
            </Button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Limit Comparison Card */}
          <Card className="border-rule shadow-sm">
            <CardHeader className="flex-row items-center justify-between pb-4">
              <CardTitle className="font-mono text-xs text-muted-foreground uppercase tracking-wide">
                Limit Comparison
              </CardTitle>
              <span className="px-3 py-1 bg-copper/10 border border-copper/30 rounded-full font-mono text-xs text-copper">
                Adjustment: ×{limitData.adjustment}
              </span>
            </CardHeader>
            <CardContent>
              {/* Bar Chart Visualization */}
              <div className="flex items-end justify-center gap-16 py-12 mb-8">
                {/* Current */}
                <div className="text-center">
                  <div className="w-32 h-48 bg-slate-blue/80 rounded-t-lg mb-4 flex items-end justify-center pb-4">
                    <span className="font-mono text-white text-sm font-semibold">${(limitData.current / 1000).toFixed(0)}k</span>
                  </div>
                  <span className="font-mono text-xs text-muted-foreground uppercase tracking-wider">Current</span>
                </div>

                {/* Recommended */}
                <div className="text-center">
                  <div className="w-32 h-64 bg-copper rounded-t-lg mb-4 flex items-end justify-center pb-4">
                    <span className="font-mono text-white text-sm font-semibold">${(limitData.recommended / 1000).toFixed(0)}k</span>
                  </div>
                  <span className="font-mono text-xs text-copper uppercase tracking-wider">Recommended</span>
                </div>
              </div>

              {/* Stats Row */}
              <div className="grid grid-cols-3 gap-6 pt-6 border-t border-rule">
                <div>
                  <div className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Delta Value</div>
                  <div className="font-mono text-xl font-bold text-ink">+${limitData.deltaValue.toLocaleString()}</div>
                </div>
                <div>
                  <div className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Risk Exposure</div>
                  <div className="font-mono text-xl font-bold text-sage">{limitData.riskExposure}</div>
                </div>
                <div>
                  <div className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Confidence</div>
                  <div className="font-mono text-xl font-bold text-copper">{limitData.confidence}%</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Adjustment Factor Breakdown */}
          <Card className="border-rule shadow-sm">
            <CardHeader className="pb-4">
              <CardTitle className="font-mono text-xs text-muted-foreground uppercase tracking-wide">
                Adjustment Factor Breakdown (Weight %)
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {adjustmentFactors.map(({ label, weight }) => (
                <div key={label}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-serif text-sm text-ink">{label}</span>
                    <span className="font-mono text-sm font-semibold text-copper">{weight}%</span>
                  </div>
                  <div className="h-2 bg-sand rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-copper rounded-full transition-all duration-500"
                      style={{ width: `${weight * 2.5}%` }}
                    />
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Insight and Loss Rate Cards */}
          <div className="grid grid-cols-2 gap-6">
            {/* Insight Card */}
            <Card className="bg-slate-blue border-0 text-white shadow-lg overflow-hidden">
              <CardContent className="pt-6 relative">
                <div className="absolute left-0 top-0 bottom-0 w-16 bg-slate-blue/50 flex items-center justify-center border-r border-white/10">
                  <Lightbulb className="h-8 w-8 text-copper" />
                </div>
                <div className="ml-20">
                  <h3 className="font-serif text-xl font-semibold text-copper mb-3">Insight: Upside Potential</h3>
                  <p className="text-white/80 text-sm leading-relaxed mb-4">
                    GLC&apos;s current utilization is at 88%. By increasing the limit to $150k, we project a 12% increase 
                    in transaction volume within 90 days with negligible risk impact.
                  </p>
                  <button className="font-mono text-xs text-white/70 hover:text-white transition-colors underline underline-offset-4">
                    READ DETAILED FORECAST
                  </button>
                </div>
              </CardContent>
            </Card>

            {/* Predicted Loss Rate Card */}
            <Card className="border-rule shadow-sm">
              <CardContent className="pt-6">
                <div className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider mb-4">
                  Predicted Loss Rate
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-mono text-4xl font-bold text-ink mb-2">0.08%</div>
                    <span className="px-2 py-1 bg-sage/15 text-sage font-mono text-xs rounded border border-sage/30">
                      ELITE GRADE
                    </span>
                  </div>
                  <div className="relative">
                    <Gauge className="h-16 w-16 text-sage" />
                    <span className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 font-mono text-xs text-sage">
                      Low
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Right Sidebar - Recommendation Factors */}
        <div>
          <Card className="border-rule shadow-sm sticky top-24">
            <CardHeader className="pb-4">
              <CardTitle className="font-mono text-xs text-muted-foreground uppercase tracking-wide">
                Recommendation Factors
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {factors.map(({ label, description, status }) => (
                <div 
                  key={label} 
                  className={`p-4 rounded-lg ${
                    status === "positive" ? "bg-sage/5 border border-sage/20" : "bg-amber/5 border border-amber/20"
                  }`}
                >
                  <div className="flex items-start gap-3">
                    {status === "positive" ? (
                      <Check className="h-5 w-5 text-sage flex-shrink-0 mt-0.5" />
                    ) : (
                      <AlertTriangle className="h-5 w-5 text-amber flex-shrink-0 mt-0.5" />
                    )}
                    <div>
                      <h4 className="font-serif text-sm font-semibold text-ink mb-1">{label}</h4>
                      <p className={`text-xs leading-relaxed ${
                        status === "positive" ? "text-sage/80" : "text-amber/80"
                      }`}>
                        {description}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Footer Timestamp */}
      <div className="flex items-center justify-center gap-4 mt-8 pt-6 border-t border-rule">
        <span className="font-mono text-xs text-muted-foreground">
          DATA REFRESHED: NOV 14, 2023 · 14:32 UTC
        </span>
        <span className="text-rule">·</span>
        <button className="font-mono text-xs text-copper hover:text-copper-dark flex items-center gap-1.5 transition-colors">
          <RefreshCw className="h-3 w-3" />
          REFRESH NOW
        </button>
      </div>
    </div>
  )
}
