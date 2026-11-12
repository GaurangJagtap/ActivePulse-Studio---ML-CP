import { Badge } from "@/components/ui/badge";
import type { ExperimentStatus } from "@/types/api";

export function StatusBadge({ status }: { status: ExperimentStatus | string }) {
  switch (status) {
    case "completed":
      return <Badge variant="success">Completed</Badge>;
    case "running":
      return (
        <Badge variant="default">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#2563EB]" />
          Running
        </Badge>
      );
    case "awaiting_annotations":
      return <Badge variant="teal">Awaiting Annotations</Badge>;
    case "failed":
      return <Badge variant="danger">Failed</Badge>;
    default:
      return <Badge variant="neutral">{status}</Badge>;
  }
}

