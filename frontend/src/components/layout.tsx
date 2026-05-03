import React, { useState } from 'react';
import { 
  LayoutDashboard, 
  UserSearch, 
  History, 
  Activity, 
  PlusCircle, 
  Bell, 
  Settings, 
  Search,
  CheckCircle2,
  LogOut,
  X,
  CreditCard,
  User,
  Info,
  HelpCircle,
  ExternalLink
} from 'lucide-react';
import { Page } from '../types';
import { motion, AnimatePresence } from 'motion/react';

interface SidebarProps {
  currentPage: Page;
  setCurrentPage: (page: Page) => void;
}

export function Sidebar({ currentPage, setCurrentPage }: SidebarProps) {
  const navItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'profiling', label: 'Profiling', icon: UserSearch },
    { id: 'history', label: 'History', icon: History },
    { id: 'api-status', label: 'API Status', icon: Activity },
  ];

  return (
    <aside className="fixed left-0 top-0 h-screen w-[240px] bg-slate-50 border-r border-rule flex flex-col py-6 space-y-4 z-50">
      <div className="px-6 mb-8 cursor-pointer" onClick={() => setCurrentPage('landing')}>
        <h1 className="font-heading text-2xl font-black text-ink mb-1">CreditIntel</h1>
        <p className="font-mono text-[10px] uppercase tracking-widest text-slate-blue opacity-70">Precision Intelligence</p>
      </div>
      <nav className="flex-1 space-y-1">
        {navItems.map((item) => {
          const isActive = currentPage === item.id || (item.id === 'profiling' && (currentPage === 'default-detail' || currentPage === 'limit-detail'));
          return (
            <button
              key={item.id}
              onClick={() => setCurrentPage(item.id as Page)}
              className={`w-full flex items-center space-x-3 px-4 py-3 transition-all cursor-pointer ${
                isActive 
                  ? 'bg-white text-copper font-bold rounded-r-lg border-l-4 border-copper shadow-sm' 
                  : 'text-slate-600 hover:bg-slate-100 rounded-lg mx-2'
              }`}
            >
              <item.icon size={20} />
              <span className="font-serif text-sm font-medium">{item.label}</span>
            </button>
          );
        })}
      </nav>
      <div className="px-4 mt-auto space-y-2">
        <button 
          onClick={() => setCurrentPage('profiling')}
          className="w-full bg-copper text-white font-mono py-3 px-4 rounded-xl shadow-sm hover:opacity-90 transition-opacity flex items-center justify-center gap-2 cursor-pointer"
        >
          <PlusCircle size={18} />
          Run New Prediction
        </button>
        <button 
          onClick={() => setCurrentPage('landing')}
          className="w-full text-slate-500 hover:text-ink font-mono py-3 px-4 rounded-xl transition-colors flex items-center justify-center gap-2 text-xs uppercase tracking-widest cursor-pointer"
        >
          <LogOut size={16} />
          Exit to Home
        </button>
      </div>
    </aside>
  );
}

interface TopNavProps {
  title: string;
}

export function TopNav({ title }: TopNavProps) {
  const [activeMenu, setActiveMenu] = useState<string | null>(null);

  const notifications = [
    { id: 1, text: 'Prediction analysis complete for #IND-9922', time: '2m ago' },
    { id: 2, text: 'Unusual activity detected in EU-WEST node', time: '1h ago' },
  ];

  return (
    <header className="flex justify-between items-center px-8 py-3 w-full bg-white border-b border-rule shadow-sm sticky top-0 z-40">
      <div className="flex items-center gap-8">
        <h2 className="text-xl font-bold text-ink font-heading tracking-tight">{title}</h2>
        <nav className="hidden md:flex space-x-6">
          <button 
            onClick={() => setActiveMenu('docs')}
            className="text-slate-500 hover:text-ink text-sm font-serif transition-colors cursor-pointer"
          >
            Documentation
          </button>
          <button 
            onClick={() => setActiveMenu('support')}
            className="text-slate-500 hover:text-ink text-sm font-serif transition-colors cursor-pointer"
          >
            Support
          </button>
        </nav>
      </div>
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2 px-3 py-1 bg-sage/10 rounded-full">
          <div className="w-2 h-2 rounded-full bg-sage animate-pulse"></div>
          <span className="font-mono text-[11px] text-sage font-bold">API: Operational</span>
        </div>
        <div className="flex items-center gap-4 border-l border-rule pl-6 relative">
          <button 
            onClick={() => setActiveMenu(activeMenu === 'notif' ? null : 'notif')}
            className={`transition-colors cursor-pointer ${activeMenu === 'notif' ? 'text-copper' : 'text-slate-500 hover:text-copper'}`}
          >
            <Bell size={20} />
            <div className="absolute top-0 right-[72px] w-2 h-2 bg-copper rounded-full border-2 border-white"></div>
          </button>
          <button 
            onClick={() => setActiveMenu(activeMenu === 'settings' ? null : 'settings')}
            className={`transition-colors cursor-pointer ${activeMenu === 'settings' ? 'text-copper' : 'text-slate-500 hover:text-copper'}`}
          >
            <Settings size={20} />
          </button>
          <button 
            onClick={() => setActiveMenu(activeMenu === 'profile' ? null : 'profile')}
            className="w-8 h-8 rounded-full bg-sand border border-rule overflow-hidden cursor-pointer hover:ring-2 hover:ring-copper transition-all"
          >
            <img 
              src="https://images.unsplash.com/photo-1560250097-0b93528c311a?auto=format&fit=crop&q=80&w=100&h=100" 
              alt="User" 
              className="w-full h-full object-cover"
            />
          </button>

          {/* Popovers */}
          <AnimatePresence>
            {activeMenu && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="absolute top-full right-0 mt-4 w-72 bg-white border border-rule shadow-xl rounded-xl p-4 z-50 text-left"
              >
                <div className="flex justify-between items-center mb-4">
                  <h4 className="font-mono text-[11px] uppercase tracking-widest text-slate-500 font-bold">
                    {activeMenu.replace(/^\w/, (c) => c.toUpperCase())}
                  </h4>
                  <button onClick={() => setActiveMenu(null)} className="text-slate-400 hover:text-ink">
                    <X size={14} />
                  </button>
                </div>
                
                {activeMenu === 'notif' && (
                  <div className="space-y-3">
                    {notifications.map(n => (
                      <div key={n.id} className="p-2 hover:bg-mist rounded-lg cursor-pointer transition-colors">
                        <p className="text-[11px] text-ink font-serif leading-snug">{n.text}</p>
                        <span className="text-[10px] font-mono text-slate-400">{n.time}</span>
                      </div>
                    ))}
                    <button className="w-full text-center py-2 font-mono text-[10px] text-copper uppercase font-bold border-t border-rule mt-2">
                      See all notifications
                    </button>
                  </div>
                )}

                {activeMenu === 'settings' && (
                  <div className="space-y-1">
                    {['Account Preference', 'Model Settings', 'API Keys', 'Team Access'].map(opt => (
                      <button key={opt} className="w-full text-left px-3 py-2 text-xs font-serif text-slate-600 hover:bg-mist hover:text-ink rounded-lg transition-all">
                        {opt}
                      </button>
                    ))}
                  </div>
                )}

                {activeMenu === 'profile' && (
                  <div className="space-y-1">
                    <div className="px-3 pb-3 border-b border-rule mb-2">
                        <p className="font-serif font-bold text-sm text-ink truncate">Stephanie Ogbogu</p>
                        <p className="font-mono text-[10px] text-slate-500">Administrator</p>
                    </div>
                    {['Manage Profile', 'Billing', 'Usage Limits'].map(opt => (
                      <button key={opt} className="w-full text-left px-3 py-2 text-xs font-serif text-slate-600 hover:bg-mist hover:text-ink rounded-lg transition-all">
                        {opt}
                      </button>
                    ))}
                    <button className="w-full text-left px-3 py-2 text-xs font-mono text-error hover:bg-red-50 rounded-lg transition-all border-t border-rule mt-2">
                      Sign Out
                    </button>
                  </div>
                )}

                {(activeMenu === 'docs' || activeMenu === 'support') && (
                  <div className="space-y-4 py-2">
                    <div className="bg-mist p-4 rounded-lg">
                      <p className="text-xs text-slate-600 font-serif leading-relaxed italic">
                        {activeMenu === 'docs' 
                          ? "Access our complete technical documentation, SDK guides, and OpenAPI schemas for institutional deployment."
                          : "Connect with our specialist support team for technical integration issues or hardware inquiries."}
                      </p>
                    </div>
                    <button onClick={() => setActiveMenu(null)} className="w-full bg-ink text-white py-2 rounded-lg font-mono text-[10px] uppercase font-bold tracking-widest flex items-center justify-center gap-2">
                      Jump to Portal <ExternalLink size={14} />
                    </button>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </header>
  );
}

export function Layout({ children, currentPage, setCurrentPage, title }: { 
  children: React.ReactNode; 
  currentPage: Page; 
  setCurrentPage: (page: Page) => void;
  title: string;
}) {
  if (currentPage === 'landing') {
    return <>{children}</>;
  }

  return (
    <div className="min-h-screen bg-mist flex">
      <Sidebar currentPage={currentPage} setCurrentPage={setCurrentPage} />
      <main className="ml-[240px] flex-1 flex flex-col min-h-screen">
        <TopNav title={title} />
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
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
