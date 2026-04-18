import { useState } from "react";
import { Activity } from "lucide-react";
import { PipelineForm } from "@/components/pipeline-form";
import { JobHistory } from "@/components/job-history";
import { PipelineProgress } from "@/components/pipeline-progress";
import { PipelineResults } from "@/components/pipeline-results";

export default function Home() {
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const handleJobStarted = (jobId: string) => {
    setActiveJobId(jobId);
  };

  const handleSelectJob = (jobId: string) => {
    setActiveJobId(jobId);
  };

  return (
    <div className="min-h-[100dvh] flex flex-col font-mono text-sm selection:bg-primary/30">
      {/* Top Header */}
      <header className="border-b border-border bg-card px-6 py-4 flex items-center justify-between sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded bg-primary/20 flex items-center justify-center text-primary">
            <Activity size={18} />
          </div>
          <div>
            <h1 className="font-bold text-base tracking-tight text-foreground">AutoML Studio</h1>
            <p className="text-xs text-muted-foreground font-sans">Automated Machine Learning Pipeline</p>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <main className="flex-1 p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 max-w-[1600px] mx-auto w-full">
        {/* Left Column: Form & History */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          <PipelineForm onJobStarted={handleJobStarted} />
          <JobHistory activeJobId={activeJobId} onSelectJob={handleSelectJob} />
        </div>

        {/* Right Column: Progress & Results */}
        <div className="lg:col-span-8 flex flex-col gap-6">
          {activeJobId ? (
            <>
              <PipelineProgress jobId={activeJobId} />
              <PipelineResults jobId={activeJobId} />
            </>
          ) : (
            <div className="flex-1 rounded-md border border-dashed border-border bg-card/30 flex flex-col items-center justify-center p-12 text-center">
              <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center text-muted-foreground mb-4">
                <Activity size={24} />
              </div>
              <h2 className="text-lg font-medium text-foreground mb-2">No Active Pipeline</h2>
              <p className="text-muted-foreground font-sans max-w-md">
                Configure and run a pipeline from the panel on the left to see progress and results here.
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
