'use client';

import { useState, useEffect } from 'react';
import { Claim, generateInitialClaims, generateNewClaim, calculateDashboardStats } from '@/lib/fraud-data';
import { StatsCards } from '@/components/stats-cards';
import { ClaimsTable } from '@/components/claims-table';
import { ClaimDetailPanel } from '@/components/claim-detail-panel';
import { Shield, Activity } from 'lucide-react';

export default function Dashboard() {
  const [claims, setClaims] = useState<Claim[]>([]);
  const [selectedClaim, setSelectedClaim] = useState<Claim | null>(null);
  const [isPanelOpen, setIsPanelOpen] = useState(false);

  // Initialize claims on mount
  useEffect(() => {
    setClaims(generateInitialClaims(12));
  }, []);

  // Auto-refresh with new claims every 3 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setClaims((prev) => {
        const newClaim = generateNewClaim();
        // Keep last 20 claims for performance
        return [newClaim, ...prev].slice(0, 20);
      });
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const handleSelectClaim = (claim: Claim) => {
    setSelectedClaim(claim);
    setIsPanelOpen(true);
  };

  const stats = calculateDashboardStats(claims);

  return (
    <main className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-white/10 bg-background/50 backdrop-blur-sm sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Shield className="w-6 h-6 text-primary" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-foreground">
                  Fraud Detection
                </h1>
                <p className="text-sm text-muted-foreground">
                  Real-time insurance claim analysis
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
              <Activity className="w-4 h-4 text-emerald-400 animate-pulse" />
              <span className="text-sm font-medium text-emerald-400">
                Live
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Stats */}
        <StatsCards
          totalProcessed={stats.totalProcessed}
          fraudDetected={stats.fraudDetected}
          averageFraudScore={stats.averageFraudScore}
        />

        {/* Claims Table Section */}
        <div>
          <div className="mb-4">
            <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
              <span className="w-1 h-6 bg-primary rounded-full" />
              Live Claims
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              Click on any claim to view detailed fraud assessment
            </p>
          </div>
          <ClaimsTable
            claims={claims}
            selectedClaimId={selectedClaim?.id}
            onSelectClaim={handleSelectClaim}
          />
        </div>
      </div>

      {/* Detail Panel */}
      <ClaimDetailPanel
        claim={selectedClaim}
        isOpen={isPanelOpen}
        onClose={() => {
          setIsPanelOpen(false);
          setSelectedClaim(null);
        }}
      />
    </main>
  );
}
