'use client';

import { TrendingUp, AlertTriangle, BarChart3 } from 'lucide-react';

interface StatsCardsProps {
  totalProcessed: number;
  fraudDetected: number;
  averageFraudScore: number;
}

export function StatsCards({
  totalProcessed,
  fraudDetected,
  averageFraudScore,
}: StatsCardsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      <div className="stat-card group hover:bg-white/8 transition-colors duration-300">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm font-medium text-muted-foreground mb-2">
              Total Claims Processed
            </p>
            <p className="text-3xl font-bold text-primary mb-1">
              {totalProcessed.toLocaleString()}
            </p>
            <p className="text-xs text-muted-foreground">Last 24 hours</p>
          </div>
          <div className="p-3 rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
            <BarChart3 className="w-5 h-5 text-primary" />
          </div>
        </div>
        <div className="absolute inset-0 bg-gradient-to-br from-primary/0 via-transparent to-primary/0 opacity-0 group-hover:opacity-10 transition-opacity rounded-lg" />
      </div>

      <div className="stat-card group hover:bg-white/8 transition-colors duration-300">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm font-medium text-muted-foreground mb-2">
              Fraud Detected
            </p>
            <p className="text-3xl font-bold text-red-400 mb-1">
              {fraudDetected}
            </p>
            <p className="text-xs text-muted-foreground">
              {((fraudDetected / totalProcessed) * 100).toFixed(1)}% of claims
            </p>
          </div>
          <div className="p-3 rounded-lg bg-red-500/10 group-hover:bg-red-500/20 transition-colors">
            <AlertTriangle className="w-5 h-5 text-red-400" />
          </div>
        </div>
        <div className="absolute inset-0 bg-gradient-to-br from-red-500/0 via-transparent to-red-500/0 opacity-0 group-hover:opacity-10 transition-opacity rounded-lg" />
      </div>

      <div className="stat-card group hover:bg-white/8 transition-colors duration-300">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm font-medium text-muted-foreground mb-2">
              Avg Fraud Score
            </p>
            <p className="text-3xl font-bold text-primary mb-1">
              {averageFraudScore.toFixed(1)}
            </p>
            <p className="text-xs text-muted-foreground">Based on {totalProcessed} claims</p>
          </div>
          <div className="p-3 rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
            <TrendingUp className="w-5 h-5 text-primary" />
          </div>
        </div>
        <div className="absolute inset-0 bg-gradient-to-br from-primary/0 via-transparent to-primary/0 opacity-0 group-hover:opacity-10 transition-opacity rounded-lg" />
      </div>
    </div>
  );
}
