import { useGetPipelineStatus, getGetPipelineStatusQueryKey } from "@workspace/api-client-react";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Loader2, TerminalSquare, AlertTriangle, CheckCircle2 } from "lucide-react";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";

interface PipelineProgressProps {
  jobId: string;
}

export function PipelineProgress({ jobId }: PipelineProgressProps) {
  const { data: job, isLoading, error } = useGetPipelineStatus(jobId, {
    query: {
      enabled: !!jobId,
      queryKey: getGetPipelineStatusQueryKey(jobId),
      refetchInterval: (query) => {
        // Poll every 2s if status is pending or running
        const status = query.state.data?.status;
        return status === "pending" || status === "running" ? 2000 : false;
      },
    }
  });

  if (isLoading) {
    return (
      <Card className="border-border bg-card">
        <CardContent className="p-6 flex items-center gap-4 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin text-primary" />
          <span className="font-sans">Fetching job status...</span>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive" className="bg-destructive/10 border-destructive/20 text-destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Error fetching status</AlertTitle>
        <AlertDescription className="font-sans">
          Could not load the status for job {jobId}.
        </AlertDescription>
      </Alert>
    );
  }

  if (!job) return null;

  // Don't show progress block if completed successfully, Results will be shown
  if (job.status === "completed") {
    return (
      <Card className="border-emerald-500/30 bg-emerald-500/5">
        <CardContent className="p-4 flex items-center gap-3 text-emerald-500">
          <CheckCircle2 className="h-5 w-5" />
          <span className="font-medium">Pipeline completed successfully</span>
        </CardContent>
      </Card>
    );
  }

  if (job.status === "failed") {
    return (
      <Alert variant="destructive" className="bg-destructive/10 border-destructive/20 text-destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Pipeline Execution Failed</AlertTitle>
        <AlertDescription className="font-sans whitespace-pre-wrap font-mono text-xs mt-2 bg-black/20 p-3 rounded border border-destructive/20">
          {job.error || "Unknown error occurred during pipeline execution."}
        </AlertDescription>
      </Alert>
    );
  }

  // Calculate approximate progress based on stage text parsing (hacky but works for UI)
  let progressValue = 10;
  const stage = job.stage?.toLowerCase() || "";
  
  if (stage.includes("preprocess") || stage.includes("clean")) progressValue = 30;
  if (stage.includes("feature")) progressValue = 50;
  if (stage.includes("train") || stage.includes("model")) progressValue = 70;
  if (stage.includes("eval") || stage.includes("valid") || stage.includes("score")) progressValue = 90;
  if (stage.includes("finish") || stage.includes("done")) progressValue = 100;

  return (
    <Card className="border-border bg-card overflow-hidden">
      <div className="bg-muted px-4 py-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2 text-foreground font-medium text-sm">
          <TerminalSquare className="h-4 w-4 text-primary" />
          Execution Log
        </div>
        <div className="text-xs text-muted-foreground font-mono opacity-70">
          {job.jobId}
        </div>
      </div>
      <CardContent className="p-6">
        <div className="mb-6">
          <div className="flex justify-between items-end mb-2">
            <span className="text-sm font-medium text-foreground">Current Stage</span>
            <span className="text-xs text-primary font-mono">{progressValue}%</span>
          </div>
          <Progress value={progressValue} className="h-2 bg-muted">
            <div 
              className="h-full bg-primary transition-all duration-500 ease-in-out" 
              style={{ width: `${progressValue}%` }}
            />
          </Progress>
        </div>

        <div className="bg-black/50 border border-border rounded-md p-4 font-mono text-sm">
          <div className="flex items-center gap-3 text-primary mb-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="animate-pulse">Processing...</span>
          </div>
          <div className="text-muted-foreground border-l-2 border-primary/50 pl-3 ml-1 py-1">
            {job.stage || "Initializing pipeline components..."}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
