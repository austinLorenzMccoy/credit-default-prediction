"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Download, GitCompare, TrendingUp, Sparkles, Users, TrendingDown, CheckCircle2 } from "lucide-react"
import Link from "next/link"

interface PageProps {
  params: Promise<{ id: string }>
}

export default function DefaultRiskDetailPage({ params }: PageProps) {
  const riskData = {
    probability: 15,
    riskLevel: "LOW RISK",
    parametersUsed: {
      ageOfEntity: "25 Years",
      creditLimit: "$100,000",
      maritalStatus: "Married",
      educationLevel: "Graduate",
      recentCreditInquiry: "None detected",
    },
    riskFactors: [
      { 
        label: "Low pay/bill ratio", 
        description: "Payment activity has decreased by 12% over the last fiscal quarter.",
        type: "warning"
      },
      { 
        label: "Consistent Liquidity", 
        description: "Cash reserves have maintained a 3:1 ratio relative to current debt obligations.",
        type: "positive"
      },
    ],
    historicalContext: {
      marketConfidence: 92.4,
      modelVariance: 1.2,
      peerBenchmark: "Top 5%",
    },
    paymentHistory: [
      { period: "SEP 2023", billAmount: 12400.00, paidAmount: 12400.00, status: "SETTLED", variance: "0.0%" },
      { period: "AUG 2023", billAmount: 11200.00, paidAmount: 10000.00, status: "PARTIAL", variance: "-10.7%" },
      { period: "JUL 2023", billAmount: 14500.00, paidAmount: 14500.00, status: "SETTLED", variance: "0.0%" },
    ],
  }

  return (
    <div className="max-w-7xl mx-auto">
      {/* Breadcrumb and Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 font-mono text-xs text-muted-foreground mb-2">
          <Link href="/dashboard" className="hover:text-copper transition-colors">DASHBOARD</Link>
          <span className="text-rule">›</span>
          <span className="text-copper">PREDICTION ANALYSIS</span>
        </div>

        <div className="flex items-center justify-between">
          <div>
            <h1 className="font-serif text-3xl font-bold text-ink">Default Risk Detail: #INT-99283</h1>
          </div>
          <div className="flex items-center gap-3">
            <Button variant="outline" className="font-mono text-sm border-copper text-copper hover:bg-copper/10">
              <Download className="h-4 w-4 mr-2" />
              EXPORT REPORT
            </Button>
            <Button className="font-mono text-sm bg-copper hover:bg-copper-dark text-white">
              <GitCompare className="h-4 w-4 mr-2" />
              COMPARE VERSIONS
            </Button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Probability of Default Card */}
          <Card className="border-rule shadow-sm">
            <CardContent className="pt-8 pb-6">
              <div className="flex items-center justify-between mb-8">
                <div className="font-mono text-xs text-muted-foreground uppercase tracking-wide">
                  Probability of Default
                </div>
                <span className="px-3 py-1 bg-sage/15 border border-sage/30 rounded-full font-mono text-xs text-sage">
                  {riskData.riskLevel}
                </span>
              </div>

              {/* Risk Gauge */}
              <div className="flex flex-col items-center py-8">
                <svg viewBox="0 0 200 130" className="w-72 h-44">
                  {/* Background arc */}
                  <path
                    d="M 20 110 A 80 80 0 0 1 180 110"
                    fill="none"
                    stroke="#D9D0C7"
                    strokeWidth="14"
                    strokeLinecap="round"
                  />
                  {/* Green segment (0-30%) */}
                  <path
                    d="M 20 110 A 80 80 0 0 1 62 40"
                    fill="none"
                    stroke="#4A7C59"
                    strokeWidth="14"
                    strokeLinecap="round"
                  />
                  {/* Amber segment (30-70%) */}
                  <path
                    d="M 62 40 A 80 80 0 0 1 138 40"
                    fill="none"
                    stroke="#D4A017"
                    strokeWidth="14"
                    strokeLinecap="round"
                  />
                  {/* Red segment (70-100%) */}
                  <path
                    d="M 138 40 A 80 80 0 0 1 180 110"
                    fill="none"
                    stroke="#C4622D"
                    strokeWidth="14"
                    strokeLinecap="round"
                  />
                  {/* Needle */}
                  <line
                    x1="100"
                    y1="110"
                    x2={100 + Math.cos((180 - riskData.probability * 1.8) * Math.PI / 180) * 60}
                    y2={110 - Math.sin((180 - riskData.probability * 1.8) * Math.PI / 180) * 60}
                    stroke="#0D1B2A"
                    strokeWidth="3"
                    strokeLinecap="round"
                  />
                  <circle cx="100" cy="110" r="8" fill="#0D1B2A" />
                </svg>

                <div className="font-mono text-5xl font-bold text-ink mt-4 mb-4">
                  {riskData.probability}%
                </div>
                <p className="text-center text-muted-foreground max-w-md">
                  The predictive model estimates a high likelihood of repayment based on current capital liquidity and historical bill ratios.
                </p>
              </div>

              {/* Risk Legend */}
              <div className="flex items-center justify-center gap-8 pt-6 border-t border-rule">
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-sage" />
                  <span className="font-mono text-xs text-muted-foreground">LOW (0-30%)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-amber" />
                  <span className="font-mono text-xs text-muted-foreground">MEDIUM (31-70%)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-copper" />
                  <span className="font-mono text-xs text-muted-foreground">HIGH (71-100%)</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Historical Prediction Context */}
          <Card className="border-rule shadow-sm">
            <CardHeader className="flex-row items-center justify-between pb-4">
              <CardTitle className="font-mono text-xs text-muted-foreground uppercase tracking-wide">
                Historical Prediction Context
              </CardTitle>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs text-muted-foreground">VIEW MODE:</span>
                <div className="flex border border-rule rounded-md overflow-hidden">
                  <button className="px-3 py-1.5 bg-sand font-mono text-xs text-ink">GRID</button>
                  <button className="px-3 py-1.5 font-mono text-xs text-muted-foreground hover:bg-sand/50 transition-colors">LIST</button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-sand/50 rounded-lg p-4 border border-rule">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Market Confidence</span>
                    <TrendingUp className="h-4 w-4 text-sage" />
                  </div>
                  <div className="font-mono text-2xl font-bold text-ink mb-1">{riskData.historicalContext.marketConfidence}%</div>
                  <div className="h-1.5 bg-sage/30 rounded-full">
                    <div className="h-full bg-sage rounded-full" style={{ width: `${riskData.historicalContext.marketConfidence}%` }} />
                  </div>
                </div>
                <div className="bg-sand/50 rounded-lg p-4 border border-rule">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Model Variance</span>
                    <Sparkles className="h-4 w-4 text-copper" />
                  </div>
                  <div className="font-mono text-2xl font-bold text-ink mb-1">{riskData.historicalContext.modelVariance}%</div>
                  <div className="h-1.5 bg-copper/30 rounded-full">
                    <div className="h-full bg-copper rounded-full" style={{ width: `${riskData.historicalContext.modelVariance * 20}%` }} />
                  </div>
                </div>
                <div className="bg-sand/50 rounded-lg p-4 border border-rule">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Peer Benchmark</span>
                    <Users className="h-4 w-4 text-slate-blue" />
                  </div>
                  <div className="font-mono text-2xl font-bold text-ink mb-1">{riskData.historicalContext.peerBenchmark}</div>
                  <div className="h-1.5 bg-slate-blue/30 rounded-full">
                    <div className="h-full bg-slate-blue rounded-full" style={{ width: "95%" }} />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Bill Payment History */}
          <Card className="border-rule shadow-sm">
            <CardHeader className="pb-4">
              <CardTitle className="font-mono text-xs text-muted-foreground uppercase tracking-wide">
                Bill Payment History (Raw Data)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <table className="w-full">
                <thead>
                  <tr className="border-b border-rule">
                    <th className="text-left py-3 font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Period</th>
                    <th className="text-left py-3 font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Bill Amount</th>
                    <th className="text-left py-3 font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Paid Amount</th>
                    <th className="text-left py-3 font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Status</th>
                    <th className="text-right py-3 font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Variance</th>
                  </tr>
                </thead>
                <tbody>
                  {riskData.paymentHistory.map((row) => (
                    <tr key={row.period} className="border-b border-rule last:border-0">
                      <td className="py-4 font-mono text-sm text-ink">{row.period}</td>
                      <td className="py-4 font-mono text-sm text-ink">${row.billAmount.toLocaleString('en-US', { minimumFractionDigits: 2 })}</td>
                      <td className="py-4 font-mono text-sm text-ink">${row.paidAmount.toLocaleString('en-US', { minimumFractionDigits: 2 })}</td>
                      <td className="py-4">
                        <span className={`px-2 py-1 rounded font-mono text-[10px] ${
                          row.status === "SETTLED" 
                            ? "bg-sand text-muted-foreground" 
                            : "bg-amber/15 text-amber border border-amber/30"
                        }`}>
                          {row.status}
                        </span>
                      </td>
                      <td className={`py-4 text-right font-mono text-sm ${
                        row.variance === "0.0%" ? "text-sage" : "text-copper"
                      }`}>
                        {row.variance}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </div>

        {/* Right Sidebar */}
        <div className="space-y-6">
          {/* Parameters Used */}
          <Card className="border-rule shadow-sm">
            <CardHeader className="pb-4">
              <div className="flex items-center gap-2">
                <div className="h-5 w-5 flex items-center justify-center">
                  <svg viewBox="0 0 20 20" className="h-4 w-4 text-muted-foreground">
                    <rect x="3" y="3" width="5" height="5" fill="currentColor" rx="1" />
                    <rect x="12" y="3" width="5" height="5" fill="currentColor" rx="1" />
                    <rect x="3" y="12" width="5" height="5" fill="currentColor" rx="1" />
                    <rect x="12" y="12" width="5" height="5" fill="currentColor" rx="1" />
                  </svg>
                </div>
                <CardTitle className="font-mono text-xs text-muted-foreground uppercase tracking-wide">
                  Parameters Used
                </CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {Object.entries(riskData.parametersUsed).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between py-1">
                  <span className="font-mono text-xs text-copper capitalize">
                    {key.replace(/([A-Z])/g, ' $1').trim()}
                  </span>
                  <span className="font-serif text-sm text-ink font-medium">{value}</span>
                </div>
              ))}
              <div className="mt-4 p-3 bg-sage/10 rounded-lg border border-sage/20">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-sage" />
                  <div>
                    <div className="font-mono text-xs text-copper">Recent Credit Inquiry</div>
                    <div className="font-serif text-sm text-sage">{riskData.parametersUsed.recentCreditInquiry}</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Risk Factors */}
          <Card className="border-rule shadow-sm">
            <CardHeader className="pb-4">
              <div className="flex items-center gap-2">
                <AlertTriangleIcon className="h-4 w-4 text-muted-foreground" />
                <CardTitle className="font-mono text-xs text-muted-foreground uppercase tracking-wide">
                  Risk Factors
                </CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {riskData.riskFactors.map(({ label, description, type }) => (
                <div 
                  key={label} 
                  className={`p-4 rounded-lg ${
                    type === "positive" ? "bg-sage/10 border border-sage/20" : "bg-copper/10 border border-copper/20"
                  }`}
                >
                  <div className="flex items-start gap-3">
                    {type === "positive" ? (
                      <CheckCircle2 className="h-5 w-5 text-sage flex-shrink-0 mt-0.5" />
                    ) : (
                      <TrendingDown className="h-5 w-5 text-copper flex-shrink-0 mt-0.5" />
                    )}
                    <div>
                      <h4 className="font-serif text-sm font-semibold text-ink mb-1">{label}</h4>
                      <p className={`text-xs leading-relaxed ${
                        type === "positive" ? "text-sage/80" : "text-copper/80"
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

      {/* Footer */}
      <footer className="flex items-center justify-between mt-12 pt-6 border-t border-rule">
        <div className="flex items-center gap-6">
          <span className="font-serif font-bold text-ink">CreditIntel</span>
          <span className="font-mono text-xs text-muted-foreground uppercase tracking-wider">Confidence in Every Data Point</span>
        </div>
        <div className="flex items-center gap-6">
          <Link href="#" className="font-mono text-xs text-muted-foreground hover:text-copper transition-colors">PRIVACY POLICY</Link>
          <Link href="#" className="font-mono text-xs text-muted-foreground hover:text-copper transition-colors">TERMS OF SERVICE</Link>
          <Link href="#" className="font-mono text-xs text-muted-foreground hover:text-copper transition-colors">AUDIT LOGS</Link>
        </div>
      </footer>
    </div>
  )
}

function AlertTriangleIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>
      <path d="M12 9v4"/>
      <path d="M12 17h.01"/>
    </svg>
  )
}
