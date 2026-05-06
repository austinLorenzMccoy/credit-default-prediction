"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Download, Search, SlidersHorizontal, TrendingUp, CheckSquare, Clock, ChevronLeft, ChevronRight } from "lucide-react"
import Link from "next/link"

interface PredictionEntry {
  date: string
  entityName: string
  entityId: string
  type: "CREDIT LIMIT" | "DEFAULT RISK"
  riskPercent: number | null
  limit: number | null
  status: "APPROVED" | "HIGH RISK" | "RE-EVALUATE" | "MODERATE"
}

const predictions: PredictionEntry[] = [
  { date: "Oct 24, 2023", entityName: "Global Tech Solutions", entityId: "GTS-4492-B", type: "CREDIT LIMIT", riskPercent: 12.4, limit: 450000, status: "APPROVED" },
  { date: "Oct 23, 2023", entityName: "Redwood Logistics", entityId: "RLX-8812-A", type: "DEFAULT RISK", riskPercent: 78.2, limit: null, status: "HIGH RISK" },
  { date: "Oct 22, 2023", entityName: "Alpine Ventures", entityId: "ALP-0023-F", type: "CREDIT LIMIT", riskPercent: 44.1, limit: 125000, status: "RE-EVALUATE" },
  { date: "Oct 22, 2023", entityName: "Inland Capital Group", entityId: "ICG-1209-C", type: "CREDIT LIMIT", riskPercent: 8.5, limit: 2500000, status: "APPROVED" },
  { date: "Oct 21, 2023", entityName: "Zion Pharma", entityId: "ZPH-3341-X", type: "DEFAULT RISK", riskPercent: 32.9, limit: null, status: "MODERATE" },
]

export default function HistoryPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [typeFilter, setTypeFilter] = useState("all")
  const [riskFilter, setRiskFilter] = useState("all")
  const [currentPage, setCurrentPage] = useState(1)

  const totalPredictions = 1482

  const getStatusColor = (status: string) => {
    switch (status) {
      case "APPROVED":
        return "bg-sand text-muted-foreground border border-rule"
      case "HIGH RISK":
        return "bg-copper/15 text-copper border border-copper/30"
      case "RE-EVALUATE":
        return "bg-amber/15 text-amber border border-amber/30"
      case "MODERATE":
        return "bg-amber/15 text-amber border border-amber/30"
      default:
        return "bg-sand text-muted-foreground"
    }
  }

  const getRiskColor = (risk: number | null) => {
    if (risk === null) return "text-muted-foreground"
    if (risk <= 30) return "text-sage"
    if (risk <= 60) return "text-amber"
    return "text-copper"
  }

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="font-serif text-3xl font-bold text-ink mb-2">Prediction History</h1>
          <p className="font-sans text-muted-foreground">
            A comprehensive log of all risk assessments and credit limit simulations.
          </p>
        </div>
        <Button variant="outline" className="font-mono text-sm border-rule hover:border-copper hover:text-copper">
          <Download className="h-4 w-4 mr-2" />
          Export CSV
        </Button>
      </div>

      {/* Filters and Stats Row */}
      <div className="flex items-start gap-6 mb-6">
        {/* Filters Card */}
        <Card className="flex-1 border-rule shadow-sm">
          <CardContent className="pt-6">
            <div className="grid grid-cols-4 gap-4 items-end">
              {/* Search */}
              <div>
                <label className="font-mono text-xs text-muted-foreground uppercase tracking-wide mb-2 block">
                  Search by Client
                </label>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="ID or Entity name..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10 border-rule focus-visible:ring-copper"
                  />
                </div>
              </div>

              {/* Type Filter */}
              <div>
                <label className="font-mono text-xs text-muted-foreground uppercase tracking-wide mb-2 block">
                  Type
                </label>
                <Select value={typeFilter} onValueChange={setTypeFilter}>
                  <SelectTrigger className="border-rule focus:ring-copper">
                    <SelectValue placeholder="All Types" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    <SelectItem value="credit-limit">Credit Limit</SelectItem>
                    <SelectItem value="default-risk">Default Risk</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Risk Level Filter */}
              <div>
                <label className="font-mono text-xs text-muted-foreground uppercase tracking-wide mb-2 block">
                  Risk Level
                </label>
                <Select value={riskFilter} onValueChange={setRiskFilter}>
                  <SelectTrigger className="border-rule focus:ring-copper">
                    <SelectValue placeholder="All Risks" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Risks</SelectItem>
                    <SelectItem value="low">Low (0-30%)</SelectItem>
                    <SelectItem value="medium">Medium (31-60%)</SelectItem>
                    <SelectItem value="high">High (61-100%)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Filter Button */}
              <Button variant="ghost" className="text-muted-foreground hover:text-copper">
                <SlidersHorizontal className="h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Total Predictions Badge */}
        <Card className="bg-slate-blue border-0 text-white shadow-lg min-w-[180px]">
          <CardContent className="pt-6">
            <div className="font-mono text-[10px] text-white/60 uppercase tracking-wider mb-1">
              Total Predictions
            </div>
            <div className="font-mono text-3xl font-bold">{totalPredictions.toLocaleString()}</div>
          </CardContent>
        </Card>
      </div>

      {/* Data Table */}
      <Card className="border-rule shadow-sm mb-6">
        <CardContent className="p-0">
          <table className="w-full">
            <thead>
              <tr className="border-b border-rule bg-sand/30">
                <th className="text-left py-4 px-6 font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Date</th>
                <th className="text-left py-4 px-6 font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Entity / ID</th>
                <th className="text-left py-4 px-6 font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Type</th>
                <th className="text-left py-4 px-6 font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Risk %</th>
                <th className="text-left py-4 px-6 font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Limit</th>
                <th className="text-left py-4 px-6 font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Status</th>
                <th className="text-right py-4 px-6 font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((prediction, index) => (
                <tr key={index} className="border-b border-rule last:border-0 hover:bg-sand/20 transition-colors">
                  <td className="py-4 px-6">
                    <span className="font-mono text-sm text-muted-foreground">{prediction.date}</span>
                  </td>
                  <td className="py-4 px-6">
                    <div>
                      <div className="font-serif text-sm font-medium text-ink">{prediction.entityName}</div>
                      <div className="font-mono text-xs text-muted-foreground">ID: {prediction.entityId}</div>
                    </div>
                  </td>
                  <td className="py-4 px-6">
                    <span className={`px-2 py-1 rounded font-mono text-[10px] ${
                      prediction.type === "CREDIT LIMIT" 
                        ? "bg-slate-blue/15 text-slate-blue border border-slate-blue/30"
                        : "bg-copper/15 text-copper border border-copper/30"
                    }`}>
                      {prediction.type}
                    </span>
                  </td>
                  <td className="py-4 px-6">
                    <span className={`font-mono text-lg font-semibold ${getRiskColor(prediction.riskPercent)}`}>
                      {prediction.riskPercent !== null ? `${prediction.riskPercent}%` : "—"}
                    </span>
                  </td>
                  <td className="py-4 px-6">
                    <span className="font-mono text-sm text-ink">
                      {prediction.limit !== null ? `$${prediction.limit.toLocaleString()}` : "—"}
                    </span>
                  </td>
                  <td className="py-4 px-6">
                    <span className={`px-3 py-1 rounded-full font-mono text-[10px] ${getStatusColor(prediction.status)}`}>
                      {prediction.status}
                    </span>
                  </td>
                  <td className="py-4 px-6 text-right">
                    <Link 
                      href={`/dashboard/profiling/${prediction.entityId}`}
                      className="font-mono text-xs text-copper hover:text-copper-dark transition-colors"
                    >
                      View<br/>Report
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {/* Pagination */}
      <div className="flex items-center justify-between mb-8">
        <span className="font-mono text-sm text-muted-foreground">
          Showing 1 to 5 of {totalPredictions.toLocaleString()} entries
        </span>
        <div className="flex items-center gap-2">
          <Button 
            variant="outline" 
            size="icon"
            className="border-rule h-9 w-9"
            disabled={currentPage === 1}
            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          {[1, 2, 3].map(page => (
            <Button
              key={page}
              variant={currentPage === page ? "default" : "outline"}
              className={`h-9 w-9 font-mono text-sm ${
                currentPage === page 
                  ? "bg-slate-blue hover:bg-slate-blue/90 text-white" 
                  : "border-rule hover:border-copper hover:text-copper"
              }`}
              onClick={() => setCurrentPage(page)}
            >
              {page}
            </Button>
          ))}
          <span className="font-mono text-sm text-muted-foreground px-2">...</span>
          <Button
            variant="outline"
            className="h-9 w-auto px-3 font-mono text-sm border-rule hover:border-copper hover:text-copper"
            onClick={() => setCurrentPage(297)}
          >
            297
          </Button>
          <Button 
            variant="outline" 
            size="icon"
            className="border-rule h-9 w-9"
            onClick={() => setCurrentPage(p => Math.min(297, p + 1))}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Bottom Stats Cards */}
      <div className="grid grid-cols-3 gap-6">
        {/* Volume Trend */}
        <Card className="border-rule shadow-sm">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-copper" />
                <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Volume Trend</span>
              </div>
            </div>
            {/* Simple bar chart */}
            <div className="flex items-end justify-between h-24 gap-2 mb-4">
              {[40, 50, 45, 90].map((height, i) => (
                <div 
                  key={i} 
                  className={`flex-1 rounded-t ${i === 3 ? 'bg-copper' : 'bg-sand-dark'}`}
                  style={{ height: `${height}%` }}
                />
              ))}
            </div>
            <div className="text-sage font-mono text-sm">
              +14% vs previous week
            </div>
          </CardContent>
        </Card>

        {/* Approval Rate */}
        <Card className="border-rule shadow-sm">
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 mb-4">
              <CheckSquare className="h-4 w-4 text-copper" />
              <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Approval Rate</span>
            </div>
            <div className="font-mono text-4xl font-bold text-copper mb-2">68.2%</div>
            <div className="text-muted-foreground text-sm">
              Consistent with Q3 target
            </div>
          </CardContent>
        </Card>

        {/* Avg Response */}
        <Card className="border-rule shadow-sm">
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 mb-4">
              <Clock className="h-4 w-4 text-copper" />
              <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Avg Response</span>
            </div>
            <div className="font-mono text-4xl font-bold text-ink mb-2">1.2s</div>
            <div className="text-muted-foreground text-sm">
              Real-time inference performance
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Footer */}
      <footer className="flex items-center justify-center mt-12 pt-6 border-t border-rule">
        <span className="font-mono text-xs text-muted-foreground">
          © 2023 CREDITINTEL PRO · PROPRIETARY ALGORITHM V4.2.1
        </span>
      </footer>
    </div>
  )
}
