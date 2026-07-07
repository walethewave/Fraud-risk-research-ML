import "server-only";
import discoveryData from "@/data/discovery.json";

export type CandidateFeature = {
  feature: string;
  fnMean: number;
  tpMean: number;
  tnMean: number;
  anomalyZ: number;
};

export type DiscoveryData = {
  falseNegativeAnalysis: {
    nFalseNegatives: number;
    nTruePositives: number;
    nTrueNegatives: number;
    candidates: CandidateFeature[];
  };
  llmProposal: {
    modelUsed: string;
    rawResponse: string;
    recommendedFeatures: string[];
  };
  impact: {
    candidateFeatures: string[];
    baselineAuc: number;
    baselineRecall: number;
    baselineFalseNegatives: number;
    augmentedAuc: number;
    augmentedRecall: number;
    augmentedFalseNegatives: number;
    aucDelta: number;
    recallDelta: number;
    nNewlyCaught: number;
    nNewlyMissed: number;
    verdict: "IMPROVED" | "NO MEANINGFUL CHANGE" | "WORSE";
  };
};

export function getDiscoveryData(): DiscoveryData {
  return discoveryData as DiscoveryData;
}
