'use client';

import { Claim } from '@/lib/fraud-data';
import { ChevronRight } from 'lucide-react';

interface ClaimsTableProps {
  claims: Claim[];
  selectedClaimId?: string;
  onSelectClaim: (claim: Claim) => void;
}

function getFraudScoreColor(score: number): string {
  if (score >= 70) return 'text-red-400 font-semibold';
  if (score >= 45) return 'text-yellow-400 font-semibold';
  return 'text-emerald-400 font-semibold';
}

function getFraudScoreBgColor(score: number): string {
  if (score >= 70) return 'bg-red-500/10';
  if (score >= 45) return 'bg-yellow-500/10';
  return 'bg-emerald-500/10';
}

export function ClaimsTable({
  claims,
  selectedClaimId,
  onSelectClaim,
}: ClaimsTableProps) {
  return (
    <div className="card-glass overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/5 bg-white/2">
              <th className="px-6 py-4 text-left font-semibold text-muted-foreground">
                Claim ID
              </th>
              <th className="px-6 py-4 text-left font-semibold text-muted-foreground">
                Amount
              </th>
              <th className="px-6 py-4 text-left font-semibold text-muted-foreground">
                Policy Age
              </th>
              <th className="px-6 py-4 text-left font-semibold text-muted-foreground">
                Hospital
              </th>
              <th className="px-6 py-4 text-left font-semibold text-muted-foreground">
                Fraud Score
              </th>
              <th className="px-6 py-4 text-center font-semibold text-muted-foreground">
                Action
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {claims.map((claim) => (
              <tr
                key={claim.id}
                className={`hover:bg-white/5 transition-colors cursor-pointer group ${
                  selectedClaimId === claim.id ? 'bg-white/10' : ''
                }`}
                onClick={() => onSelectClaim(claim)}
              >
                <td className="px-6 py-4">
                  <code className="text-xs font-mono text-primary">
                    {claim.id}
                  </code>
                </td>
                <td className="px-6 py-4 font-semibold">
                  ${claim.amount.toLocaleString()}
                </td>
                <td className="px-6 py-4">{claim.policyAge} years</td>
                <td className="px-6 py-4 text-muted-foreground">
                  {claim.hospital}
                </td>
                <td className="px-6 py-4">
                  <div
                    className={`inline-flex items-center justify-center w-12 h-8 rounded-md font-semibold text-sm ${getFraudScoreBgColor(
                      claim.fraudScore
                    )} ${getFraudScoreColor(claim.fraudScore)}`}
                  >
                    {Math.round(claim.fraudScore)}
                  </div>
                </td>
                <td className="px-6 py-4 text-center">
                  <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-foreground group-hover:translate-x-1 transition-all" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
