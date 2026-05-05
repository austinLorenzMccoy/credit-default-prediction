import React, { useState } from 'react';
import {
  LayoutDashboard, UserSearch, History, Activity,
  PlusCircle, Bell, Settings, X, LogOut, ExternalLink,
} from 'lucide-react';
import { Page } from '../types';
import { motion, AnimatePresence } from 'motion/react';

interface SidebarProps {
  currentPage: Page;
  setCurrentPage: (page: Page) => void;
}

export function Sidebar({ currentPage, setCurrentPage }: SidebarProps) {
  const navItems = [
    { id: 'overview',  label: 'Overview',  icon: LayoutDashboard },
    { id: 'profiling', label: 'Profiling', icon: UserSearch },
    { id: 'history',   label: 'History',   icon: History },
    { id: 'api-status',label: 'API Status', icon: Activity },
  ];

  return (
    <aside className="fixed left-0 top-0 h-screen w-[240px] bg-white border-r border-rule flex flex-col py-6 z-50">
      {/* Brand */}
      <div className="px-6 mb-8 cursor-pointer flex items-center gap-2.5" onClick={() => setCurrentPage('landing')}>
        <div className="w-7 h-7 rounded-md bg-gradient-to-br from-copper to-slate-blue flex items-center justify-center flex-shrink-0">
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
            <path d="M3 12L8 4L13 12H3Z" fill="white" fillOpacity="0.9"/>
          </svg>
        </div>
        <div>
          <h1 className="font-heading text-lg text-ink leading-none">CreditIntel</h1>
          <p className="font-mono text-[9px] uppercase tracking-widest text-slate-blue/60 mt-0.5">Precision Intelligence</p>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 space-y-1 px-3">
        {navItems.map(item => {
          const isActive =
            currentPage === item.id ||
            (item.id === 'profiling' && (currentPage === 'default-detail' || currentPage === 'limit-detail'));
          return (
            <button
              key={item.id}
              onClick={() => setCurrentPage(item.id as Page)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all cursor-pointer ${
                isActive
                  ? 'bg-copper/08 text-copper font-bold border-l-[3px] border-copper'
                  : 'text-slate-600 hover:bg-mist hover:text-ink border-l-[3px] border-transparent'
              }`}
            >
              <item.icon size={18} />
              <span className="font-body text-sm">{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Bottom actions */}
      <div className="px-4 mt-auto space-y-2">
        <button
          onClick={() => setCurrentPage('profiling')}
          className="w-full bg-copper text-white font-mono text-[10px] font-bold uppercase tracking-widest py-3 px-4 rounded-xl shadow-ambient hover:bg-copper-dk transition-colors flex items-center justify-center gap-2"
        >
          <PlusCircle size={16} /> Run New Prediction
        </button>
        <button
          onClick={() => setCurrentPage('landing')}
          className="w-full text-slate-400 hover:text-ink font-mono text-[10px] uppercase tracking-widest py-2.5 px-4 rounded-xl transition-colors flex items-center justify-center gap-2"
        >
          <LogOut size={14} /> Exit to Home
        </button>
      </div>
    </aside>
  );
}

// ── Initials avatar (no CDN dependency) ─────────────────────────────────
function Avatar({ name }: { name: string }) {
  const initials = name.split(' ').map(n => n[0]).join('').slice(0, 2).toUpperCase();
  return (
    <div className="w-8 h-8 rounded-full bg-slate-blue text-white font-mono text-xs font-bold flex items-center justify-center flex-shrink-0 hover:ring-2 hover:ring-copper transition-all cursor-pointer">
      {initials}
    </div>
  );
}

interface TopNavProps {
  title: string;
}

export function TopNav({ title }: TopNavProps) {
  const [activeMenu, setActiveMenu] = useState<string | null>(null);
  const userName = 'Stephanie Ogbogu';

  const notifications = [
    { id: 1, text: 'Prediction analysis complete for #IND-9922', time: '2m ago' },
    { id: 2, text: 'Unusual activity detected in EU-WEST node', time: '1h ago' },
  ];

  return (
    <header className="flex justify-between items-center px-8 py-3 w-full bg-white border-b border-rule shadow-ambient sticky top-0 z-40">
      <div className="flex items-center gap-8">
        <h2 className="font-heading text-xl text-ink tracking-tight">{title}</h2>
        <nav className="hidden md:flex gap-6">
          {['Documentation', 'Support'].map(label => (
            <button
              key={label}
              onClick={() => setActiveMenu(activeMenu === label.toLowerCase() ? null : label.toLowerCase())}
              className="font-body text-sm text-slate-500 hover:text-ink transition-colors cursor-pointer"
            >
              {label}
            </button>
          ))}
        </nav>
      </div>

      <div className="flex items-center gap-5 relative">
        {/* API status pill */}
        <div className="flex items-center gap-1.5 px-3 py-1 bg-sage/10 rounded-full border border-sage/20">
          <div className="w-1.5 h-1.5 rounded-full bg-sage animate-pulse" />
          <span className="font-mono text-[10px] text-sage font-bold uppercase tracking-wide">API: Operational</span>
        </div>

        {/* Icons */}
        <div className="flex items-center gap-3 border-l border-rule pl-5">
          <button
            onClick={() => setActiveMenu(activeMenu === 'notif' ? null : 'notif')}
            className={`relative transition-colors ${activeMenu === 'notif' ? 'text-copper' : 'text-slate-400 hover:text-copper'}`}
          >
            <Bell size={18} />
            <div className="absolute -top-1 -right-1 w-2 h-2 bg-copper rounded-full border-2 border-white" />
          </button>
          <button
            onClick={() => setActiveMenu(activeMenu === 'settings' ? null : 'settings')}
            className={`transition-colors ${activeMenu === 'settings' ? 'text-copper' : 'text-slate-400 hover:text-copper'}`}
          >
            <Settings size={18} />
          </button>
          <div onClick={() => setActiveMenu(activeMenu === 'profile' ? null : 'profile')}>
            <Avatar name={userName} />
          </div>
        </div>

        {/* Popovers */}
        <AnimatePresence>
          {activeMenu && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 8 }}
              className="absolute top-full right-0 mt-4 w-72 bg-white border border-rule shadow-lifted rounded-xl p-4 z-50"
            >
              <div className="flex justify-between items-center mb-4">
                <h4 className="font-mono text-[10px] uppercase tracking-widest text-slate-500 font-bold capitalize">
                  {activeMenu}
                </h4>
                <button onClick={() => setActiveMenu(null)} className="text-slate-400 hover:text-ink">
                  <X size={14} />
                </button>
              </div>

              {activeMenu === 'notif' && (
                <div className="space-y-2">
                  {notifications.map(n => (
                    <div key={n.id} className="p-2.5 hover:bg-mist rounded-lg cursor-pointer transition-colors">
                      <p className="font-body text-xs text-ink leading-snug">{n.text}</p>
                      <span className="font-mono text-[10px] text-slate-400">{n.time}</span>
                    </div>
                  ))}
                  <button className="w-full text-center py-2 font-mono text-[10px] text-copper uppercase font-bold border-t border-rule mt-1">
                    See all notifications
                  </button>
                </div>
              )}

              {activeMenu === 'settings' && (
                <div className="space-y-1">
                  {['Account Preference', 'Model Settings', 'API Keys', 'Team Access'].map(opt => (
                    <button key={opt} className="w-full text-left px-3 py-2 font-body text-xs text-slate-600 hover:bg-mist hover:text-ink rounded-lg transition-all">
                      {opt}
                    </button>
                  ))}
                </div>
              )}

              {activeMenu === 'profile' && (
                <div>
                  <div className="flex items-center gap-3 px-3 pb-4 border-b border-rule mb-2">
                    <Avatar name={userName} />
                    <div>
                      <p className="font-heading text-sm text-ink">{userName}</p>
                      <p className="font-mono text-[10px] text-slate-500">Administrator</p>
                    </div>
                  </div>
                  {['Manage Profile', 'Billing', 'Usage Limits'].map(opt => (
                    <button key={opt} className="w-full text-left px-3 py-2 font-body text-xs text-slate-600 hover:bg-mist rounded-lg transition-all">
                      {opt}
                    </button>
                  ))}
                  <button className="w-full text-left px-3 py-2 font-mono text-xs text-error hover:bg-red-50 rounded-lg transition-all border-t border-rule mt-2">
                    Sign Out
                  </button>
                </div>
              )}

              {(activeMenu === 'documentation' || activeMenu === 'support') && (
                <div className="space-y-4">
                  <div className="bg-mist p-4 rounded-lg">
                    <p className="font-body text-xs text-slate-600 leading-relaxed italic">
                      {activeMenu === 'documentation'
                        ? 'Complete technical documentation, SDK guides, and OpenAPI schemas for institutional deployment.'
                        : 'Connect with our specialist support team for technical integration issues or enquiries.'}
                    </p>
                  </div>
                  <button className="w-full bg-ink text-white py-2 rounded-lg font-mono text-[10px] uppercase font-bold tracking-widest flex items-center justify-center gap-2">
                    Open Portal <ExternalLink size={13} />
                  </button>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </header>
  );
}

export function Layout({
  children, currentPage, setCurrentPage, title,
}: {
  children: React.ReactNode;
  currentPage: Page;
  setCurrentPage: (page: Page) => void;
  title: string;
}) {
  if (currentPage === 'landing') return <>{children}</>;

  return (
    <div className="min-h-screen bg-mist flex">
      <Sidebar currentPage={currentPage} setCurrentPage={setCurrentPage} />
      <main className="ml-[240px] flex-1 flex flex-col min-h-screen">
        <TopNav title={title} />
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="p-10 max-w-7xl mx-auto w-full"
        >
          {children}
        </motion.div>
      </main>
    </div>
  );
}
