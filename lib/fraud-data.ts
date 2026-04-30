export interface Claim {
  id: string;
  amount: number;
  policyAge: number;
  hospital: string;
  fraudScore: number;
  timestamp: Date;
}

export interface ClaimDetail extends Claim {
  decision: 'High Risk' | 'Low Risk';
  explanation: string;
  evidenceCases: string[];
  riskFactors: string[];
}

const hospitals = [
  'Metropolitan Hospital',
  'St. Mary\'s Medical Center',
  'Harbor View Hospital',
  'Riverside Medical',
  'United Health Center',
  'Downtown Clinic',
  'Central Medical Complex',
  'Northside Hospital',
];

const explanations = {
  high: [
    'Claim submitted 2 days after hospital visit with unusually high treatment costs. Similar pattern detected in 47 fraud cases.',
    'Multiple procedures billed by same provider within 24 hours. Matches signature of known fraud ring identified in Q3.',
    'Patient claims duplicate procedures at different facilities. 89% correlation with flagged fraud cases from past 6 months.',
    'Amount exceeds policy limit by 35%. Inconsistent with verified patient treatment history.',
    'Treatment duration 5x longer than typical for diagnosis code. Associated with 94% of identified fraud cases.',
  ],
  low: [
    'Claim matches standard treatment protocol for diagnosis. Verified against 500+ legitimate cases.',
    'All procedures documented in hospital records with physician signatures. No red flags detected.',
    'Billing amounts consistent with regional averages. Patient has clean 8-year claim history.',
    'Multiple cross-checks passed: facility credentials verified, diagnosis code valid, timeline consistent.',
    'Routine follow-up procedure with established patient. Low-risk pattern based on historical data.',
  ],
};

const evidenceCases = [
  'CASE-2024-001: Duplicate billing fraud, recovery $45,000',
  'CASE-2024-015: Phantom treatment claim, prevented $78,500',
  'CASE-2024-042: Collision of care fraud, detected $33,200',
  'CASE-2023-089: Billing code manipulation, recovery $62,100',
  'CASE-2023-156: Provider collusion scheme, prevented $156,000',
];

function generateFraudScore(): number {
  // Weighted distribution: more low scores, some medium, fewer high
  const rand = Math.random();
  if (rand < 0.65) {
    return Math.random() * 30 + Math.random() * 20; // 0-50
  } else if (rand < 0.85) {
    return Math.random() * 30 + 50; // 50-80
  } else {
    return Math.random() * 20 + 80; // 80-100
  }
}

function generateClaimData(): Claim {
  const baseId = Math.random().toString(36).substring(7).toUpperCase();
  const amount = Math.floor(Math.random() * 45000) + 5000;
  const policyAge = Math.floor(Math.random() * 15) + 1;
  const hospital = hospitals[Math.floor(Math.random() * hospitals.length)];
  const fraudScore = generateFraudScore();

  return {
    id: `CLM-${Date.now()}-${baseId}`,
    amount,
    policyAge,
    hospital,
    fraudScore,
    timestamp: new Date(),
  };
}

export function getDetailedClaimInfo(claim: Claim): ClaimDetail {
  const isHighRisk = claim.fraudScore > 65;
  const decision = isHighRisk ? 'High Risk' : 'Low Risk';
  const explanationList = isHighRisk ? explanations.high : explanations.low;
  const explanation = explanationList[Math.floor(Math.random() * explanationList.length)];

  const selectedEvidence = [];
  for (let i = 0; i < 3; i++) {
    const idx = Math.floor(Math.random() * evidenceCases.length);
    if (!selectedEvidence.includes(evidenceCases[idx])) {
      selectedEvidence.push(evidenceCases[idx]);
    }
  }

  const riskFactors = [];
  if (claim.fraudScore > 75) {
    riskFactors.push('High billing amount');
    riskFactors.push('Unusual provider pattern');
    riskFactors.push('Multiple procedures in short timeframe');
  } else if (claim.fraudScore > 50) {
    riskFactors.push('Moderate billing amount');
    riskFactors.push('Standard provider pattern');
  } else {
    riskFactors.push('Verified treatment codes');
    riskFactors.push('Consistent billing patterns');
  }

  return {
    ...claim,
    decision,
    explanation,
    evidenceCases: selectedEvidence,
    riskFactors,
  };
}

export function generateInitialClaims(count: number): Claim[] {
  return Array.from({ length: count }, () => generateClaimData());
}

export function generateNewClaim(): Claim {
  return generateClaimData();
}

export function calculateDashboardStats(claims: Claim[]) {
  const fraudCount = claims.filter((c) => c.fraudScore > 65).length;
  const avgScore =
    claims.reduce((sum, c) => sum + c.fraudScore, 0) / claims.length;

  return {
    totalProcessed: claims.length,
    fraudDetected: fraudCount,
    averageFraudScore: avgScore,
  };
}
