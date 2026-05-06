"use client"

import { Bell, Settings, Search } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import Image from "next/image"

interface DashboardHeaderProps {
  showSearch?: boolean
  title?: string
}

export function DashboardHeader({ showSearch = true, title }: DashboardHeaderProps) {
  return (
    <header className="h-16 border-b border-rule bg-white flex items-center justify-between px-6 sticky top-0 z-10">
      <div className="flex items-center gap-4">
        {showSearch && (
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input 
              placeholder="Search customer records..." 
              className="w-[280px] pl-10 bg-mist border-rule focus-visible:ring-copper"
            />
          </div>
        )}
        {title && (
          <div className="flex items-center gap-3">
            <span className="font-serif text-lg font-semibold text-ink">{title}</span>
          </div>
        )}
      </div>

      <div className="flex items-center gap-4">
        {/* API Status */}
        <div className="flex items-center gap-2 px-3 py-1.5 bg-sage/10 border border-sage/30 rounded-full">
          <span className="h-2 w-2 bg-sage rounded-full animate-pulse" />
          <span className="font-mono text-xs text-sage">API: Operational</span>
        </div>

        {/* Notifications */}
        <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-ink">
          <Bell className="h-5 w-5" />
        </Button>

        {/* Settings */}
        <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-ink">
          <Settings className="h-5 w-5" />
        </Button>

        {/* User Avatar */}
        <div className="h-9 w-9 rounded-full bg-slate-blue overflow-hidden">
          <Image 
            src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=36&h=36&fit=crop&crop=face"
            alt="User avatar"
            width={36}
            height={36}
            className="h-full w-full object-cover"
          />
        </div>
      </div>
    </header>
  )
}
