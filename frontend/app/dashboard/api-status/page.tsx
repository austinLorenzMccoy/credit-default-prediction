"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { CheckCircle2, AlertCircle, Clock, Zap, Database, Shield } from "lucide-react"

const endpoints = [
  { path: "/api/v1/health", method: "GET", status: "operational", latency: "45ms" },
  { path: "/api/v1/predict/default", method: "POST", status: "operational", latency: "142ms" },
  { path: "/api/v1/predict/credit-limit", method: "POST", status: "operational", latency: "156ms" },
  { path: "/api/v1/batch/upload", method: "POST", status: "operational", latency: "890ms" },
  { path: "/api/v1/reports/export", method: "GET", status: "operational", latency: "234ms" },
]

const systemMetrics = [
  { label: "Uptime", value: "99.97%", icon: Clock, color: "sage" },
  { label: "Avg Latency", value: "142ms", icon: Zap, color: "copper" },
  { label: "DB Connections", value: "24/100", icon: Database, color: "slate-blue" },
  { label: "SSL Certificate", value: "Valid", icon: Shield, color: "sage" },
]

export default function ApiStatusPage() {
  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="font-serif text-3xl font-bold text-ink mb-2">API Status</h1>
        <p className="font-sans text-muted-foreground">
          Real-time health monitoring for all CreditIntel API endpoints.
        </p>
      </div>

      {/* Overall Status Banner */}
      <Card className="bg-sage/10 border-sage/30 mb-8">
        <CardContent className="pt-6">
          <div className="flex items-center gap-4">
            <div className="h-12 w-12 rounded-full bg-sage/20 flex items-center justify-center">
              <CheckCircle2 className="h-6 w-6 text-sage" />
            </div>
            <div>
              <h2 className="font-serif text-xl font-semibold text-ink">All Systems Operational</h2>
              <p className="font-mono text-sm text-sage">Last checked: Just now</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* System Metrics */}
      <div className="grid grid-cols-4 gap-6 mb-8">
        {systemMetrics.map(({ label, value, icon: Icon, color }) => (
          <Card key={label} className="border-rule shadow-sm">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between mb-4">
                <Icon className={`h-5 w-5 ${
                  color === "sage" ? "text-sage" : color === "copper" ? "text-copper" : "text-slate-blue"
                }`} />
                <span className={`px-2 py-0.5 rounded-full font-mono text-[10px] ${
                  color === "sage" ? "bg-sage/15 text-sage" : color === "copper" ? "bg-copper/15 text-copper" : "bg-slate-blue/15 text-slate-blue"
                }`}>
                  HEALTHY
                </span>
              </div>
              <div className="font-mono text-2xl font-bold text-ink mb-1">{value}</div>
              <div className="font-mono text-xs text-muted-foreground uppercase tracking-wider">{label}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Endpoints Table */}
      <Card className="border-rule shadow-sm">
        <CardHeader className="pb-4">
          <CardTitle className="font-mono text-xs text-muted-foreground uppercase tracking-wide">
            Endpoint Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <table className="w-full">
            <thead>
              <tr className="border-b border-rule">
                <th className="text-left py-3 font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Endpoint</th>
                <th className="text-left py-3 font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Method</th>
                <th className="text-left py-3 font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Status</th>
                <th className="text-right py-3 font-mono text-[10px] text-muted-foreground uppercase tracking-wider">Latency</th>
              </tr>
            </thead>
            <tbody>
              {endpoints.map((endpoint) => (
                <tr key={endpoint.path} className="border-b border-rule last:border-0">
                  <td className="py-4">
                    <span className="font-mono text-sm text-ink">{endpoint.path}</span>
                  </td>
                  <td className="py-4">
                    <span className={`px-2 py-1 rounded font-mono text-[10px] ${
                      endpoint.method === "GET" 
                        ? "bg-sage/15 text-sage border border-sage/30"
                        : "bg-copper/15 text-copper border border-copper/30"
                    }`}>
                      {endpoint.method}
                    </span>
                  </td>
                  <td className="py-4">
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-2 rounded-full bg-sage animate-pulse" />
                      <span className="font-mono text-xs text-sage capitalize">{endpoint.status}</span>
                    </div>
                  </td>
                  <td className="py-4 text-right">
                    <span className="font-mono text-sm text-muted-foreground">{endpoint.latency}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {/* Recent Incidents */}
      <Card className="border-rule shadow-sm mt-8">
        <CardHeader className="pb-4">
          <CardTitle className="font-mono text-xs text-muted-foreground uppercase tracking-wide">
            Recent Incidents (Last 30 Days)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-12 text-muted-foreground">
            <div className="text-center">
              <CheckCircle2 className="h-12 w-12 text-sage mx-auto mb-4 opacity-50" />
              <p className="font-sans text-sm">No incidents reported in the last 30 days</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
