/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Page } from './types';
import { Layout } from './components/layout';
import { LandingPage } from './components/LandingPage';
import { Dashboard } from './components/Dashboard';
import { Overview } from './components/Overview';
import { DefaultRiskDetail } from './components/DefaultRiskDetail';
import { PredictionHistory } from './components/PredictionHistory';
import { CreditLimitDetail } from './components/CreditLimitDetail';

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>('landing');

  const getPageTitle = (page: Page): string => {
    switch (page) {
      case 'overview': return 'Executive Overview';
      case 'profiling': return 'Customer Intelligence';
      case 'history': return 'Prediction Archive';
      case 'api-status': return 'System Performance';
      case 'default-detail': return 'Analysis Detail';
      case 'limit-detail': return 'Recommendation Detail';
      default: return 'CreditIntel Pro';
    }
  };

  const renderContent = () => {
    switch (currentPage) {
      case 'landing':
        return <LandingPage onStart={() => setCurrentPage('overview')} />;
      case 'overview':
        return <Overview onNavigate={(p) => setCurrentPage(p)} />;
      case 'profiling':
        return <Dashboard onPredict={(p) => setCurrentPage(p)} />;
      case 'default-detail':
        return <DefaultRiskDetail />;
      case 'history':
        return <PredictionHistory />;
      case 'limit-detail':
        return <CreditLimitDetail />;
      case 'api-status':
        return (
          <div className="flex flex-col items-center justify-center min-h-[60vh] text-center p-8 border border-dashed border-rule rounded-2xl bg-white/50">
            <h3 className="text-2xl font-heading text-ink mb-4">API Monitor</h3>
            <p className="text-slate-500 mb-8 max-w-md">Our global infrastructure is currently operating within normal parameters.</p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 w-full max-w-4xl">
              {['US-EAST', 'EU-WEST', 'ASIA-PAC', 'SA-EAST'].map(node => (
                <div key={node} className="p-4 bg-white rounded-xl shadow-ambient border border-rule">
                  <div className="font-mono text-[10px] text-slate-400 mb-2">{node}</div>
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-sage">ONLINE</span>
                    <div className="w-2 h-2 rounded-full bg-sage animate-pulse"></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      default:
        return <div>Page not found</div>;
    }
  };

  return (
    <Layout 
      currentPage={currentPage} 
      setCurrentPage={setCurrentPage} 
      title={getPageTitle(currentPage)}
    >
      {renderContent()}
    </Layout>
  );
}

