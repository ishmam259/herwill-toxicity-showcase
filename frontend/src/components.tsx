import type { ReactNode } from "react";
import { ShieldCheck } from "lucide-react";
export function Badge({ label }: { label: number }) {
  return (
    <span className={`badge class-${label}`}>
      <i />
      {["Explicit", "Subtle", "Neutral"][label]}
    </span>
  );
}
export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return <section className={`card ${className}`}>{children}</section>;
}
export function EvidenceNote() {
  return (
    <div className="evidence-note">
      <ShieldCheck size={17} />
      <p>
        <strong>Read the scores with context.</strong> Historical OOF
        diagnostics were repeatedly inspected; they are not a fresh held-out
        evaluation. All comparisons use raw argmax. Bangla specialists have
        narrower coverage.
      </p>
    </div>
  );
}
