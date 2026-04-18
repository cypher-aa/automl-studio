import { Clock, CheckCircle2, XCircle, Loader2, AlertCircle } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useListPipelineJobs, getListPipelineJobsQueryKey } from "@workspace/api-client-react";
import { formatDistanceToNow } from "date-fns";

interface JobHistoryProps {
  activeJobId: string | null;
  onSelectJob: (jobId: string) => void;
}

export function JobHistory({ activeJobId, onSelectJob }: JobHistoryProps) {
  const { data: jobs, isLoading, error } = useListPipelineJobs({
    query: {
      queryKey: getListPipelineJobsQueryKey(),
      refetchInterval: 5000, // Poll list every 5s to catch status updates from background jobs
    }
  });

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 className="h-4 w-4 text-emerald-500" />;
      case "failed":
        return <XCircle className="h-4 w-4 text-destructive" />;
      case "running":
        return <Loader2 className="h-4 w-4 text-primary animate-spin" />;
      default:
        return <Clock className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return <Badge variant="outline" className="bg-emerald-500/10 text-emerald-500 border-emerald-500/20">Done</Badge>;
      case "failed":
        return <Badge variant="outline" className="bg-destructive/10 text-destructive border-destructive/20">Failed</Badge>;
      case "running":
        return <Badge variant="outline" className="bg-primary/10 text-primary border-primary/20">Running</Badge>;
      default:
        return <Badge variant="outline" className="text-muted-foreground">Pending</Badge>;
    }
  };

  return (
    <Card className="border-border bg-card shadow-md flex-1 flex flex-col min-h-[300px]">
      <CardHeader className="pb-3 border-b border-border/50">
        <CardTitle className="text-sm font-semibold flex justify-between items-center text-muted-foreground uppercase tracking-wider">
          <span>Recent Runs</span>
          <Badge variant="secondary" className="font-mono text-xs">{jobs?.length || 0}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0 flex-1 overflow-auto">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center h-full p-8 text-muted-foreground">
            <Loader2 className="h-6 w-6 animate-spin mb-2" />
            <span className="text-sm font-sans">Loading history...</span>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-full p-8 text-destructive">
            <AlertCircle className="h-6 w-6 mb-2" />
            <span className="text-sm font-sans">Failed to load jobs</span>
          </div>
        ) : jobs && jobs.length > 0 ? (
          <div className="divide-y divide-border/50">
            {jobs.map((job) => (
              <button
                key={job.jobId}
                onClick={() => onSelectJob(job.jobId)}
                className={`w-full text-left p-4 hover:bg-muted/50 transition-colors flex flex-col gap-2 relative ${
                  activeJobId === job.jobId ? 'bg-muted/80 before:absolute before:left-0 before:top-0 before:bottom-0 before:w-1 before:bg-primary' : ''
                }`}
                data-testid={`btn-select-job-${job.jobId}`}
              >
                <div className="flex items-center justify-between w-full">
                  <div className="flex items-center gap-2 font-medium text-sm truncate pr-2 text-foreground">
                    {getStatusIcon(job.status)}
                    <span className="truncate">{job.datasetName}</span>
                  </div>
                  {getStatusBadge(job.status)}
                </div>
                
                <div className="flex items-center justify-between w-full text-xs text-muted-foreground font-mono">
                  <span className="truncate max-w-[140px] opacity-70" title={job.jobId}>{job.jobId.split('-')[0]}...</span>
                  <span>{formatDistanceToNow(new Date(job.createdAt), { addSuffix: true })}</span>
                </div>
              </button>
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full p-8 text-muted-foreground text-center">
            <Clock className="h-8 w-8 mb-3 opacity-20" />
            <span className="text-sm font-sans mb-1 text-foreground">No pipeline runs yet</span>
            <span className="text-xs">Start a new pipeline to see it here</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
