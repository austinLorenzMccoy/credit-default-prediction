"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { CheckCircle2, AlertTriangle, ArrowRight, TrendingUp, LayoutGrid } from "lucide-react"

interface PredictionResult {
  defaultRisk: number
  riskLevel: "LOW RISK" | "MEDIUM RISK" | "HIGH RISK"
  recommendedLimit: number
  adjustment: number
  liquidityScore: number
}

export default function DashboardPage() {
  const [formData, setFormData] = useState({
    age: "34",
    creditLimit: "150000",
    gender: "female",
    payStatus: "paid",
    education: "graduate",
    billAmount: "45200",
    maritalStatus: "married",
    paymentAmount: "10000",
  })

  const [result, setResult] = useState<PredictionResult | null>({
    defaultRisk: 15,
    riskLevel: "LOW RISK",
    recommendedLimit: 150000,
    adjustment: 50,
    liquidityScore: 88,
  })

  const [sessionId] = useState("CI-99284")

  const handleCalculate = () => {
    // Simulated prediction result
    const riskPercent = Math.floor(Math.random() * 30) + 5
    setResult({
      defaultRisk: riskPercent,
      riskLevel: riskPercent <= 30 ? "LOW RISK" : riskPercent <= 60 ? "MEDIUM RISK" : "HIGH RISK",
      recommendedLimit: parseInt(formData.creditLimit) * 1.5,
      adjustment: 50,
      liquidityScore: Math.floor(Math.random() * 20) + 80,
    })
  }

  const clearForm = () => {
    setFormData({
      age: "",
      creditLimit: "",
      gender: "",
      payStatus: "",
      education: "",
      billAmount: "",
      maritalStatus: "",
      paymentAmount: "",
    })
    setResult(null)
  }

  return (
    <div className="max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="mb-8">
        <h1 className="font-serif text-3xl font-bold text-ink mb-2">Customer Intelligence</h1>
        <p className="font-sans text-muted-foreground">
          Execute precision risk modeling for high-value credit applications.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Input Form */}
        <div className="lg:col-span-2">
          <Card className="border-rule shadow-sm">
            <CardHeader className="border-b border-rule pb-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="h-8 w-8 rounded bg-sand flex items-center justify-center">
                    <LayoutGrid className="h-4 w-4 text-copper" />
                  </div>
                  <div>
                    <CardTitle className="font-serif text-lg">INPUT FORM — Customer Financial Profile</CardTitle>
                  </div>
                </div>
                <div className="font-mono text-xs text-muted-foreground">
                  SESSION ID: {sessionId}
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="grid grid-cols-2 gap-6">
                {/* Left column inputs */}
                <div className="space-y-4">
                  <div>
                    <Label className="font-mono text-xs text-copper uppercase tracking-wide mb-2 block">Age</Label>
                    <Input
                      type="number"
                      value={formData.age}
                      onChange={(e) => setFormData({ ...formData, age: e.target.value })}
                      className="border-rule focus-visible:ring-copper"
                      placeholder="Enter age"
                    />
                  </div>

                  <div>
                    <Label className="font-mono text-xs text-copper uppercase tracking-wide mb-2 block">Gender</Label>
                    <Select value={formData.gender} onValueChange={(v) => setFormData({ ...formData, gender: v })}>
                      <SelectTrigger className="border-rule focus:ring-copper">
                        <SelectValue placeholder="Select gender" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="male">Male</SelectItem>
                        <SelectItem value="female">Female</SelectItem>
                        <SelectItem value="other">Other</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label className="font-mono text-xs text-copper uppercase tracking-wide mb-2 block">Education</Label>
                    <Select value={formData.education} onValueChange={(v) => setFormData({ ...formData, education: v })}>
                      <SelectTrigger className="border-rule focus:ring-copper">
                        <SelectValue placeholder="Select education" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="graduate">Graduate School</SelectItem>
                        <SelectItem value="university">University</SelectItem>
                        <SelectItem value="highschool">High School</SelectItem>
                        <SelectItem value="other">Other</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label className="font-mono text-xs text-copper uppercase tracking-wide mb-2 block">Marital Status</Label>
                    <Select value={formData.maritalStatus} onValueChange={(v) => setFormData({ ...formData, maritalStatus: v })}>
                      <SelectTrigger className="border-rule focus:ring-copper">
                        <SelectValue placeholder="Select status" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="married">Married</SelectItem>
                        <SelectItem value="single">Single</SelectItem>
                        <SelectItem value="divorced">Divorced</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {/* Right column inputs */}
                <div className="space-y-4">
                  <div>
                    <Label className="font-mono text-xs text-copper uppercase tracking-wide mb-2 block">Desired Credit Limit</Label>
                    <div className="relative">
                      <span className="absolute left-3 top-1/2 -translate-y-1/2 font-mono text-sm text-copper">N</span>
                      <Input
                        type="number"
                        value={formData.creditLimit}
                        onChange={(e) => setFormData({ ...formData, creditLimit: e.target.value })}
                        className="border-rule focus-visible:ring-copper pl-7"
                        placeholder="Enter amount"
                      />
                    </div>
                  </div>

                  <div>
                    <Label className="font-mono text-xs text-copper uppercase tracking-wide mb-2 block">Pay Status (Last Month)</Label>
                    <Select value={formData.payStatus} onValueChange={(v) => setFormData({ ...formData, payStatus: v })}>
                      <SelectTrigger className="border-rule focus:ring-copper">
                        <SelectValue placeholder="Select status" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="paid">Paid on Time</SelectItem>
                        <SelectItem value="late1">1 Month Late</SelectItem>
                        <SelectItem value="late2">2 Months Late</SelectItem>
                        <SelectItem value="late3">3+ Months Late</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label className="font-mono text-xs text-copper uppercase tracking-wide mb-2 block">Bill Amount</Label>
                    <div className="relative">
                      <span className="absolute left-3 top-1/2 -translate-y-1/2 font-mono text-sm text-copper">N</span>
                      <Input
                        type="number"
                        value={formData.billAmount}
                        onChange={(e) => setFormData({ ...formData, billAmount: e.target.value })}
                        className="border-rule focus-visible:ring-copper pl-7"
                        placeholder="Enter amount"
                      />
                    </div>
                  </div>

                  <div>
                    <Label className="font-mono text-xs text-copper uppercase tracking-wide mb-2 block">Payment Amount</Label>
                    <div className="relative">
                      <span className="absolute left-3 top-1/2 -translate-y-1/2 font-mono text-sm text-copper">N</span>
                      <Input
                        type="number"
                        value={formData.paymentAmount}
                        onChange={(e) => setFormData({ ...formData, paymentAmount: e.target.value })}
                        className="border-rule focus-visible:ring-copper pl-7"
                        placeholder="Enter amount"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center justify-center gap-4 pt-8 mt-6 border-t border-rule">
                <Button 
                  variant="outline" 
                  onClick={clearForm}
                  className="font-mono text-copper border-copper hover:bg-copper/10"
                >
                  Clear Profile
                </Button>
                <Button 
                  onClick={handleCalculate}
                  className="bg-copper hover:bg-copper-dark text-white font-mono px-8"
                >
                  Calculate Risk Prediction
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column - Results */}
        <div className="space-y-6">
          {/* Default Risk Card */}
          <Card className="border-rule shadow-sm">
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <CardTitle className="font-mono text-xs text-muted-foreground uppercase tracking-wide">Default Risk</CardTitle>
                {result && (
                  <span className={`px-3 py-1 rounded-full font-mono text-xs ${
                    result.riskLevel === "LOW RISK" 
                      ? "bg-sage/15 text-sage border border-sage/30" 
                      : result.riskLevel === "MEDIUM RISK"
                      ? "bg-amber/15 text-amber border border-amber/30"
                      : "bg-copper/15 text-copper border border-copper/30"
                  }`}>
                    {result.riskLevel}
                  </span>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {result ? (
                <div className="space-y-4">
                  {/* Risk Gauge */}
                  <div className="flex flex-col items-center py-4">
                    <svg viewBox="0 0 200 120" className="w-48 h-28">
                      {/* Background arc */}
                      <path
                        d="M 20 100 A 80 80 0 0 1 180 100"
                        fill="none"
                        stroke="#D9D0C7"
                        strokeWidth="12"
                        strokeLinecap="round"
                      />
                      {/* Progress arc */}
                      <path
                        d="M 20 100 A 80 80 0 0 1 180 100"
                        fill="none"
                        stroke={result.riskLevel === "LOW RISK" ? "#4A7C59" : result.riskLevel === "MEDIUM RISK" ? "#D4A017" : "#C4622D"}
                        strokeWidth="12"
                        strokeLinecap="round"
                        strokeDasharray={`${(result.defaultRisk / 100) * 251.2} 251.2`}
                      />
                      {/* Percentage text */}
                      <text x="100" y="85" textAnchor="middle" className="font-mono text-3xl font-bold fill-ink">
                        {result.defaultRisk}%
                      </text>
                    </svg>
                    <p className="text-sm text-muted-foreground text-center mt-2">
                      Probability of default based on behavioral payment trends.
                    </p>
                  </div>

                  {/* Insight */}
                  <div className="flex items-start gap-3 p-3 bg-sage/10 rounded-lg border border-sage/20">
                    <CheckCircle2 className="h-5 w-5 text-sage flex-shrink-0 mt-0.5" />
                    <p className="text-sm text-ink">Strong repayment profile detected.</p>
                  </div>
                </div>
              ) : (
                <div className="h-40 flex items-center justify-center text-muted-foreground">
                  Enter customer data to calculate risk
                </div>
              )}
            </CardContent>
          </Card>

          {/* Credit Limit Recommendation Card */}
          <Card className="border-rule shadow-sm">
            <CardHeader className="pb-4">
              <CardTitle className="font-mono text-xs text-muted-foreground uppercase tracking-wide">Recommended Credit Limit</CardTitle>
            </CardHeader>
            <CardContent>
              {result ? (
                <div className="space-y-4">
                  <div className="flex items-baseline gap-2">
                    <span className="font-mono text-3xl font-bold text-ink">
                      N{result.recommendedLimit.toLocaleString()}
                    </span>
                    <span className="flex items-center gap-1 text-sage font-mono text-sm">
                      <TrendingUp className="h-4 w-4" />
                      +{result.adjustment}%
                    </span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Adjustment from previous limit: N{(result.recommendedLimit / 1.5).toLocaleString()}
                  </p>
                  
                  {/* Liquidity Score */}
                  <div className="pt-4 border-t border-rule">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-mono text-xs text-copper uppercase tracking-wide">Liquidity Score</span>
                      <span className="font-mono text-sm font-semibold text-ink">{result.liquidityScore}/100</span>
                    </div>
                    <div className="h-2 bg-sand rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-copper rounded-full transition-all duration-500"
                        style={{ width: `${result.liquidityScore}%` }}
                      />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="h-32 flex items-center justify-center text-muted-foreground">
                  Awaiting calculation
                </div>
              )}
            </CardContent>
          </Card>

          {/* Institutional Insight Card */}
          {result && (
            <Card className="bg-slate-blue border-0 text-white shadow-lg overflow-hidden">
              <CardContent className="pt-6 relative">
                <div className="absolute right-0 bottom-0 opacity-10">
                  <CheckCircle2 className="h-32 w-32 -mb-4 -mr-4" />
                </div>
                <h3 className="font-serif text-xl font-semibold mb-3">Institutional Insight</h3>
                <p className="text-white/80 text-sm leading-relaxed mb-4">
                  This profile shows exceptional stability in &quot;Education&quot; and &quot;Pay Status&quot; sectors. 
                  We recommend immediate approval for the requested increase.
                </p>
                <button className="flex items-center gap-2 font-mono text-sm text-white/90 hover:text-white transition-colors">
                  VIEW DETAILED REPORT
                  <ArrowRight className="h-4 w-4" />
                </button>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Footer Stats */}
      <div className="grid grid-cols-3 gap-6 mt-8">
        <Card className="border-rule shadow-sm">
          <CardContent className="pt-6">
            <div className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Model Version</div>
            <div className="font-mono text-lg font-semibold text-ink">XGB-CREDIT v4.2</div>
          </CardContent>
        </Card>
        <Card className="border-rule shadow-sm">
          <CardContent className="pt-6">
            <div className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Inference Time</div>
            <div className="font-mono text-lg font-semibold text-ink">142ms</div>
          </CardContent>
        </Card>
        <Card className="border-rule shadow-sm">
          <CardContent className="pt-6">
            <div className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Last Profile Scan</div>
            <div className="font-mono text-lg font-semibold text-ink">Just Now</div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
