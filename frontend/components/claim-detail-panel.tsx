'use client';

import { useEffect, useState } from 'react';
import { ClaimDetail, getDetailedClaimInfo } from '@/lib/fraud-data';
import { X, CheckCircle, AlertCircle, AlertTriangle } from 'lucide-react';

interface ClaimDetailPanelProps {
  claim: any;
  isOpen: boolean;
  onClose: () => void;
}

export function ClaimDetailPanel({
  claim,
  isOpen,
  onClose,
}: ClaimDetailPanelProps) {
  const [detail, setDetail] = useState<ClaimDetail | null>(null);

  useEffect(() => {
    if (claim) {
      setDetail(getDetailedClaimInfo(claim));
    }
  }, [claim]);

  const getDecisionIcon = () => {
    if (!detail) return null;
    return detail.decision === 'High Risk' ? (
      <AlertTriangle className="w-5 h-5 text-red-400" />
    ) : (
      <CheckCircle className="w-5 h-5 text-emerald-400" />
    );
  };

  const getDecisionBadgeClass = () => {
    if (!detail) return '';
    return detail.decision === 'High Risk'
      ? 'badge-high-risk'
      : 'badge-low-risk';
  };

  const getGaugeColor = () => {
    if (!detail) return 'hsl(200 100% 50%)';
    if (detail.fraudScore >= 70) return 'hsl(0 84% 60%)';
    if (detail.fraudScore >= 45) return 'hsl(45 100% 50%)';
    return 'hsl(150 100% 50%)';
  };

  const gaugePercentage = detail ? (detail.fraudScore / 100) * 100 : 0;

  return (
    <>
      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 transition-opacity"
          onClick={onClose}
        />
      )}

      {/* Panel */}
      <div
        className={`fixed right-0 top-0 h-full w-full max-w-2xl bg-background border-l border-white/10 z-50 overflow-y-auto transform transition-transform duration-300 ease-out ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <div className="p-8">
          {/* Header */}
          <div className="flex items-start justify-between mb-8">
            <div>
              <p className="text-sm font-medium text-muted-foreground mb-2">
                Claim Details
              </p>
              <h2 className="text-2xl font-bold text-foreground">
                {claim?.id}
              </h2>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-muted-foreground" />
            </button>
          </div>

          {detail && (
            <>
              {/* Fraud Score Gauge */}
              <div className="mb-8 p-6 card-glass">
                <p className="text-sm font-medium text-muted-foreground mb-6">
                  Fraud Risk Assessment
                </p>

                <div className="flex items-center justify-center mb-6">
                  <div className="relative w-48 h-48">
                    {/* Gauge background */}
                    <svg
                      className="w-full h-full"
                      viewBox="0 0 200 120"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      {/* Background arc */}
                      <path
                        d="M 20 100 A 80 80 0 0 1 180 100"
                        fill="none"
                        stroke="rgba(255,255,255,0.1)"
                        strokeWidth="12"
                        strokeLinecap="round"
                      />
                      {/* Progress arc */}
                      <path
                        d="M 20 100 A 80 80 0 0 1 180 100"
                        fill="none"
                        stroke={getGaugeColor()}
                        strokeWidth="12"
                        strokeLinecap="round"
                        strokeDasharray={`${
                          (gaugePercentage / 100) * 251.2
                        } 251.2`}
                        className="transition-all duration-1000"
                      />
                    </svg>

                    {/* Center text */}
                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                      <p
                        className="text-4xl font-bold"
                        style={{ color: getGaugeColor() }}
                      >
                        {Math.round(detail.fraudScore)}
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Risk Score
                      </p>
                    </div>
                  </div>
                </div>

                {/* Decision */}
                <div className="flex justify-center">
                  <div className={`${getDecisionBadgeClass()} flex items-center gap-2`}>
                    {getDecisionIcon()}
                    {detail.decision}
                  </div>
                </div>
              </div>

              {/* AI Explanation */}
              <div className="mb-8 p-6 card-glass">
                <div className="flex items-center gap-2 mb-4">
                  <AlertCircle className="w-4 h-4 text-primary" />
                  <h3 className="font-semibold text-foreground">
                    AI Analysis
                  </h3>
                </div>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {detail.explanation}
                </p>
              </div>

              {/* Risk Factors */}
              <div className="mb-8 p-6 card-glass">
                <h3 className="font-semibold text-foreground mb-4">
                  Risk Factors
                </h3>
                <ul className="space-y-3">
                  {detail.riskFactors.map((factor, idx) => (
                    <li
                      key={idx}
                      className="flex items-start gap-3 text-sm text-muted-foreground"
                    >
                      <span className="block w-1.5 h-1.5 rounded-full bg-primary/60 mt-1.5 flex-shrink-0" />
                      {factor}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Similar Cases */}
              <div className="mb-8 p-6 card-glass">
                <h3 className="font-semibold text-foreground mb-4">
                  Similar Past Cases
                </h3>
                <div className="space-y-2">
                  {detail.evidenceCases.map((caseId, idx) => (
                    <div
                      key={idx}
                      className="p-3 rounded-md bg-white/3 border border-white/5 text-sm text-muted-foreground hover:bg-white/5 transition-colors cursor-pointer"
                    >
                      <code className="text-xs font-mono text-primary">
                        {caseId.split(':')[0]}
                      </code>
                      <span className="text-muted-foreground ml-2">
                        {caseId.split(':')[1]}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Claim Details */}
              <div className="p-6 card-glass mb-8">
                <h3 className="font-semibold text-foreground mb-4">
                  Claim Information
                </h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground text-xs uppercase tracking-wide mb-1">
                      Amount
                    </p>
                    <p className="font-semibold text-foreground">
                      ${claim.amount.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground text-xs uppercase tracking-wide mb-1">
                      Policy Age
                    </p>
                    <p className="font-semibold text-foreground">
                      {claim.policyAge} years
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground text-xs uppercase tracking-wide mb-1">
                      Hospital
                    </p>
                    <p className="font-semibold text-foreground">
                      {claim.hospital}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground text-xs uppercase tracking-wide mb-1">
                      Submitted
                    </p>
                    <p className="font-semibold text-foreground">
                      {claim.timestamp.toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
