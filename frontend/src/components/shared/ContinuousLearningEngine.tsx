import React, { useState } from "react";
import {
  BrainCircuit,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  GitCommit,
  TrendingUp,
  ShieldCheck,
  UserCheck,
  Clock,
  Sparkles,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { ContinuousLearningStatusResponse } from "@/types/api";
import { formatPct, formatShortDate } from "@/utils/format";

interface ContinuousLearningEngineProps {
  status: ContinuousLearningStatusResponse | null;
  onTriggerRetrain: () => Promise<void>;
  isRetraining: boolean;
}

export function ContinuousLearningEngine({
  status,
  onTriggerRetrain,
  isRetraining,
}: ContinuousLearningEngineProps) {
  const [retrainSuccessMsg, setRetrainSuccessMsg] = useState<string | null>(null);

  if (!status) {
    return null;
  }

  const getStatusBadge = (s: string) => {
    switch (s) {
      case "retraining_in_progress":
      case "preparing_data":
      case "evaluating_candidate":
        return <Badge variant="warning">Retraining in Progress</Badge>;
      case "model_updated_successfully":
        return <Badge variant="success">Model Updated Successfully</Badge>;
      case "candidate_rejected":
        return <Badge variant="warning">Candidate Rejected (Active Preserved)</Badge>;
      case "feedback_received":
        return <Badge variant="teal">New Feedback Received</Badge>;
      case "retraining_failed":
        return <Badge variant="danger">Retraining Failed</Badge>;
      default:
        return <Badge variant="neutral">Waiting for Human Feedback</Badge>;
    }
  };

  const deltaAcc = status.delta_metrics?.accuracy;
  const deltaF1 = status.delta_metrics?.macro_f1;
  const deltaEce = status.delta_metrics?.ece;

  const handleManualRetrain = async () => {
    try {
      await onTriggerRetrain();
      setRetrainSuccessMsg("Manual retraining cycle dispatched to PyTorch worker.");
      setTimeout(() => setRetrainSuccessMsg(null), 4000);
    } catch {
      // Handled in parent
    }
  };

  return (
    <Card tone="default" className="border-[#E6D6C6] bg-[#FFFCF7] shadow-card">
      <CardHeader className="border-b border-[#E6D6C6] pb-4">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-[#54301D] to-[#B98255] text-[#FFFCF7] shadow-sm">
              <BrainCircuit className="h-5 w-5" />
            </div>
            <div>
              <CardTitle className="text-base font-semibold text-[#38271F]">
                Continuous Learning Engine
              </CardTitle>
              <CardDescription className="text-xs text-[#806D60]">
                Automatic PyTorch self-retraining on human corrections & verified annotations
              </CardDescription>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {getStatusBadge(status.retraining_status)}
            <Button
              size="sm"
              variant="outline"
              disabled={isRetraining || status.samples_awaiting_training === 0}
              onClick={handleManualRetrain}
              className="text-xs"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isRetraining ? "animate-spin" : ""}`} />
              {isRetraining ? "Retraining..." : "Retrain Now"}
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6 pt-5">
        {/* Retraining status note */}
        <div className="flex items-center justify-between rounded-xl border border-[#E6D6C6] bg-[#FAF3EC] px-4 py-2.5 text-xs">
          <div className="flex items-center gap-2 text-[#38271F]">
            <Sparkles className="h-4 w-4 text-[#B98255]" />
            <span className="font-medium">{status.status_message}</span>
          </div>
          {status.last_successful_update && (
            <span className="text-[11px] text-[#806D60]">
              Last Update: {formatShortDate(status.last_successful_update)}
            </span>
          )}
        </div>

        {retrainSuccessMsg && (
          <div className="flex items-center gap-2 rounded-lg bg-[#EDF6F0] p-2.5 text-xs font-medium text-[#2D6A4F] border border-[#B7D5C4]">
            <CheckCircle2 className="h-4 w-4 shrink-0" />
            <span>{retrainSuccessMsg}</span>
          </div>
        )}

        {/* 1. Learning Activity & Model Versions Grid */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-5">
          <div className="rounded-xl border border-[#E6D6C6] bg-[#FFFFFF] p-3 shadow-sm">
            <div className="flex items-center gap-1.5 text-[11px] font-medium text-[#806D60]">
              <UserCheck className="h-3.5 w-3.5 text-[#B98255]" />
              Human Verified
            </div>
            <div className="mt-1 text-lg font-bold text-[#38271F]">
              {status.total_human_verified}
            </div>
            <div className="text-[10px] text-[#806D60]">
              {status.confirmed_predictions} confirmed
            </div>
          </div>

          <div className="rounded-xl border border-[#E6D6C6] bg-[#FFFFFF] p-3 shadow-sm">
            <div className="flex items-center gap-1.5 text-[11px] font-medium text-[#806D60]">
              <AlertCircle className="h-3.5 w-3.5 text-[#6B3E26]" />
              Corrections Learned
            </div>
            <div className="mt-1 text-lg font-bold text-[#6B3E26]">
              {status.newly_corrected}
            </div>
            <div className="text-[10px] text-[#806D60]">High-signal supervision</div>
          </div>

          <div className="rounded-xl border border-[#E6D6C6] bg-[#FFFFFF] p-3 shadow-sm">
            <div className="flex items-center gap-1.5 text-[11px] font-medium text-[#806D60]">
              <Clock className="h-3.5 w-3.5 text-[#B98255]" />
              Awaiting Training
            </div>
            <div className="mt-1 text-lg font-bold text-[#38271F]">
              {status.samples_awaiting_training}
            </div>
            <div className="text-[10px] text-[#806D60]">In queue for candidate</div>
          </div>

          <div className="rounded-xl border border-[#E6D6C6] bg-[#FFFFFF] p-3 shadow-sm">
            <div className="flex items-center gap-1.5 text-[11px] font-medium text-[#806D60]">
              <ShieldCheck className="h-3.5 w-3.5 text-[#2D6A4F]" />
              Total Training Pool
            </div>
            <div className="mt-1 text-lg font-bold text-[#38271F]">
              {status.total_training_samples}
            </div>
            <div className="text-[10px] text-[#806D60]">Accumulated samples</div>
          </div>

          <div className="col-span-2 sm:col-span-4 lg:col-span-1 rounded-xl border border-[#E6D6C6] bg-[#FFFFFF] p-3 shadow-sm">
            <div className="flex items-center gap-1.5 text-[11px] font-medium text-[#806D60]">
              <GitCommit className="h-3.5 w-3.5 text-[#B98255]" />
              Active Version
            </div>
            <div className="mt-1 font-mono text-sm font-bold text-[#6B3E26]">
              {status.active_model_version}
            </div>
            <div className="text-[10px] text-[#806D60]">
              Candidate: {status.latest_candidate_version || "—"}
            </div>
          </div>
        </div>

        {/* 2. Model Improvement Tracking Table */}
        <div>
          <div className="mb-2 flex items-center justify-between">
            <h4 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-[#38271F]">
              <TrendingUp className="h-3.5 w-3.5 text-[#B98255]" />
              Model Improvement Tracking
            </h4>
            <span className="text-[11px] text-[#806D60]">
              Measured on untouched holdout test partition
            </span>
          </div>

          <div className="overflow-hidden rounded-xl border border-[#E6D6C6] bg-[#FFFFFF]">
            <Table>
              <TableHeader className="bg-[#FAF3EC]">
                <TableRow>
                  <TableHead className="text-xs">Metric</TableHead>
                  <TableHead className="text-xs">Previous Model ({status.previous_model_version || "v0"})</TableHead>
                  <TableHead className="text-xs">Updated Active Model ({status.active_model_version})</TableHead>
                  <TableHead className="text-xs">Difference (Gain)</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                <TableRow>
                  <TableCell className="text-xs font-medium text-[#38271F]">Holdout Accuracy</TableCell>
                  <TableCell className="font-mono text-xs text-[#806D60]">
                    {formatPct(status.previous_metrics?.accuracy)}
                  </TableCell>
                  <TableCell className="font-mono text-xs font-semibold text-[#38271F]">
                    {formatPct(status.active_metrics.accuracy)}
                  </TableCell>
                  <TableCell className="font-mono text-xs">
                    {deltaAcc !== undefined ? (
                      <span className={deltaAcc >= 0 ? "font-semibold text-[#2D6A4F]" : "text-[#9A6218]"}>
                        {deltaAcc >= 0 ? `+${(deltaAcc * 100).toFixed(1)}%` : `${(deltaAcc * 100).toFixed(1)}%`}
                      </span>
                    ) : (
                      "—"
                    )}
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell className="text-xs font-medium text-[#38271F]">Macro-F1 Score</TableCell>
                  <TableCell className="font-mono text-xs text-[#806D60]">
                    {formatPct(status.previous_metrics?.macro_f1)}
                  </TableCell>
                  <TableCell className="font-mono text-xs font-semibold text-[#38271F]">
                    {formatPct(status.active_metrics.macro_f1)}
                  </TableCell>
                  <TableCell className="font-mono text-xs">
                    {deltaF1 !== undefined ? (
                      <span className={deltaF1 >= 0 ? "font-semibold text-[#2D6A4F]" : "text-[#9A6218]"}>
                        {deltaF1 >= 0 ? `+${(deltaF1 * 100).toFixed(1)}%` : `${(deltaF1 * 100).toFixed(1)}%`}
                      </span>
                    ) : (
                      "—"
                    )}
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell className="text-xs font-medium text-[#38271F]">Expected Calibration Error (ECE)</TableCell>
                  <TableCell className="font-mono text-xs text-[#806D60]">
                    {status.previous_metrics?.ece !== null && status.previous_metrics?.ece !== undefined
                      ? status.previous_metrics.ece.toFixed(4)
                      : "—"}
                  </TableCell>
                  <TableCell className="font-mono text-xs font-semibold text-[#38271F]">
                    {status.active_metrics.ece !== null && status.active_metrics.ece !== undefined
                      ? status.active_metrics.ece.toFixed(4)
                      : "—"}
                  </TableCell>
                  <TableCell className="font-mono text-xs">
                    {deltaEce !== undefined ? (
                      <span className={deltaEce <= 0 ? "font-semibold text-[#2D6A4F]" : "text-[#9A6218]"}>
                        {deltaEce <= 0 ? `${(deltaEce * 100).toFixed(2)}% (improved)` : `+${(deltaEce * 100).toFixed(2)}%`}
                      </span>
                    ) : (
                      "—"
                    )}
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </div>
        </div>

        {/* 3. Continuous Retraining History Log */}
        <div>
          <div className="mb-2 flex items-center justify-between">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-[#38271F]">
              Continuous Retraining History
            </h4>
            <span className="text-[11px] text-[#806D60]">Traceable candidate evaluations</span>
          </div>

          {status.history.length === 0 ? (
            <div className="rounded-xl border border-[#E6D6C6] bg-[#FFFFFF] py-6 text-center text-xs text-[#806D60]">
              No self-retraining cycles recorded yet. Annotate samples above to initiate continuous learning.
            </div>
          ) : (
            <div className="max-h-56 overflow-y-auto rounded-xl border border-[#E6D6C6] bg-[#FFFFFF]">
              <Table>
                <TableHeader className="bg-[#FAF3EC]">
                  <TableRow>
                    <TableHead className="text-xs">Event ID</TableHead>
                    <TableHead className="text-xs">Trigger</TableHead>
                    <TableHead className="text-xs">Samples</TableHead>
                    <TableHead className="text-xs">Candidate Acc</TableHead>
                    <TableHead className="text-xs">Status</TableHead>
                    <TableHead className="text-xs">Decision</TableHead>
                    <TableHead className="text-xs">Timestamp</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {status.history.map((ev) => (
                    <TableRow key={ev.event_id}>
                      <TableCell className="font-mono text-xs font-semibold text-[#6B3E26]">
                        {ev.event_id}
                      </TableCell>
                      <TableCell className="text-xs capitalize text-[#806D60]">
                        {ev.trigger_type.replace("_", " ")}
                      </TableCell>
                      <TableCell className="font-mono text-xs">
                        +{ev.samples_incorporated}
                      </TableCell>
                      <TableCell className="font-mono text-xs font-medium">
                        {formatPct(ev.candidate_metrics.accuracy)}
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant={
                            ev.status === "promoted"
                              ? "success"
                              : ev.status === "rejected"
                              ? "warning"
                              : ev.status === "training"
                              ? "teal"
                              : "neutral"
                          }
                        >
                          {ev.status}
                        </Badge>
                      </TableCell>
                      <TableCell className="max-w-[220px] truncate text-xs text-[#806D60]" title={ev.decision_reason || ""}>
                        {ev.decision_reason || "—"}
                      </TableCell>
                      <TableCell className="text-xs text-[#806D60]">
                        {formatShortDate(ev.completed_at || ev.created_at)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
