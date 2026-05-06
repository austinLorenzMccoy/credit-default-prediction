"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { LayoutGrid, Users, History, Terminal, Plus } from "lucide-react"

const navItems = [
  { label: "Overview", href: "/dashboard", icon: LayoutGrid },
  { label: "Profiling", href: "/dashboard/profiling", icon: Users },
  { label: "History", href: "/dashboard/history", icon: History },
  { label: "API Status", href: "/dashboard/api-status", icon: Terminal },
]

export function DashboardSidebar() {
  const pathname = usePathname()

  return (
    <aside className="w-[200px] border-r border-rule bg-white flex flex-col h-screen sticky top-0">
      {/* Logo */}
      <div className="p-6 border-b border-rule">
        <Link href="/" className="flex flex-col">
          <span className="font-serif text-xl font-bold text-ink tracking-tight">CreditIntel</span>
          <span className="font-mono text-[10px] text-muted-foreground tracking-[0.1em] uppercase">Precision Intelligence</span>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        <ul className="space-y-1">
          {navItems.map((item) => {
            const isActive = pathname === item.href || 
              (item.href !== "/dashboard" && pathname.startsWith(item.href))
            const Icon = item.icon
            
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-mono transition-colors",
                    isActive 
                      ? "bg-sand text-copper font-medium" 
                      : "text-muted-foreground hover:bg-sand/50 hover:text-ink"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </Link>
              </li>
            )
          })}
        </ul>
      </nav>

      {/* Run New Prediction Button */}
      <div className="p-4 border-t border-rule">
        <Link
          href="/dashboard"
          className="flex items-center justify-center gap-2 w-full px-4 py-3 bg-copper text-white rounded-md font-mono text-sm hover:bg-copper-dark transition-colors"
        >
          <Plus className="h-4 w-4" />
          Run New Prediction
        </Link>
      </div>
    </aside>
  )
}
