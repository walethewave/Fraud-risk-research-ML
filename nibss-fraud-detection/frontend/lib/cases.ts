import "server-only";
import casesData from "@/data/cases.json";
import type { FraudCase } from "./types";

const cases = casesData as FraudCase[];

export function getAllCases(): FraudCase[] {
  return cases;
}

export function getCaseById(id: number): FraudCase | undefined {
  return cases.find((c) => c.id === id);
}
