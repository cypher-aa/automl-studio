import { useGetPipelineStatus, useGetPipelineResult, getGetPipelineStatusQueryKey, getGetPipelineResultQueryKey } from "@workspace/api-client-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Trophy, Target, Hash, BarChart3, Database, Columns, SplitSquareVertical, AlertTriangle, FlaskConical } from "lucide-react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Cell } from "recharts";
import { Skeleton } from "@/components/ui/skeleton";
import { PredictionPanel } from "@/components/prediction-panel";

interface PipelineResultsProps {
  jobId: string;
}

export function PipelineResults({ jobId }: PipelineResultsProps) {
  const { data: job } = useGetPipelineStatus(jobId, {
    query: {
      enabled: !!jobId,
      queryKey: getGetPipelineStatusQueryKey(jobId),
    }
  });

  const { data: result, isLoading, error } = useGetPipelineResult(jobId, {
    query: {
      enabled: !!jobId && job?.status === "completed",
      queryKey: getGetPipelineResultQueryKey(jobId),
    }
  });

  if (job?.status !== "completed") {
    return null;
  }

  if (isLoading) {
    return <ResultsSkeleton />;
  }

  if (error || !result) {
    return null; // Handled by progress component if it's an error
  }

  // Format data for Recharts
  const featureData = result.featureImportance
    .slice()
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 10) // Top 10 features
    .map(f => ({
      name: f.feature,
      value: Number((f.importance * 100).toFixed(2)) // Convert to percentage
    }));

  const modelData = result.modelScores
    .slice()
    .sort((a, b) => b.score - a.score)
    .map(m => ({
      name: m.modelName,
      score: Number(m.score.toFixed(4)),
      time: m.trainingTime
    }));

  return (
    <div className="flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4 duration-700 fill-mode-both">
      {/* Top Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="bg-card border-border shadow-md relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-primary/10 rounded-full blur-3xl -mr-10 -mt-10 pointer-events-none" />
          <CardContent className="p-6">
            <div className="flex items-center gap-2 text-muted-foreground mb-4">
              <Trophy className="h-4 w-4 text-primary" />
              <h3 className="text-sm font-medium uppercase tracking-wider">Best Model</h3>
            </div>
            <div className="text-2xl font-bold text-foreground mb-1 truncate" title={result.bestModel}>
              {result.bestModel}
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="font-mono text-xs bg-primary/20 text-primary hover:bg-primary/30">
                {result.scoreMetric.toUpperCase()}
              </Badge>
              <span className="text-xl font-mono text-foreground">
                {result.bestScore.toFixed(4)}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border shadow-md">
          <CardContent className="p-6">
            <div className="flex items-center gap-2 text-muted-foreground mb-4">
              <Target className="h-4 w-4 text-primary" />
              <h3 className="text-sm font-medium uppercase tracking-wider">Target Objective</h3>
            </div>
            <div className="text-lg font-bold text-foreground mb-2 truncate">
              {result.targetColumn}
            </div>
            <Badge variant="outline" className="font-mono border-border text-muted-foreground">
              {result.problemType}
            </Badge>
          </CardContent>
        </Card>

        <Card className="bg-card border-border shadow-md">
          <CardContent className="p-6">
            <div className="flex items-center gap-2 text-muted-foreground mb-4">
              <Database className="h-4 w-4 text-primary" />
              <h3 className="text-sm font-medium uppercase tracking-wider">Dataset Shape</h3>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-xs text-muted-foreground mb-1 flex items-center gap-1">
                  <SplitSquareVertical className="h-3 w-3" /> Rows
                </div>
                <div className="text-lg font-mono text-foreground">{result.datasetSummary.rows.toLocaleString()}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground mb-1 flex items-center gap-1">
                  <Columns className="h-3 w-3" /> Columns
                </div>
                <div className="text-lg font-mono text-foreground">{result.datasetSummary.columns.toLocaleString()}</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* ── Class imbalance warning ── */}
      {result.classImbalanceWarning && (
        <div className="flex items-start gap-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-300 animate-in fade-in duration-500">
          <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0 text-amber-400" />
          <span className="font-sans">{result.classImbalanceWarning}</span>
        </div>
      )}

      {/* ── Evaluation Metrics ── */}
      <Card className="bg-card border-border shadow-md">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2 uppercase tracking-wider text-muted-foreground">
            <FlaskConical className="h-4 w-4 text-primary" />
            Evaluation Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          {result.problemType === "classification" ? (
            <div className="flex flex-col gap-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {[
                  { label: "Accuracy", value: result.bestScore, highlight: !!result.classImbalanceWarning },
                  { label: "F1 Score", value: result.f1Score ?? null, highlight: !!result.classImbalanceWarning },
                  { label: "Precision", value: result.precision ?? null, highlight: false },
                  { label: "Recall", value: result.recall ?? null, highlight: false },
                ].map(({ label, value, highlight }) => (
                  <div key={label} className={`p-3 rounded-md border text-center ${highlight ? "border-amber-500/30 bg-amber-500/5" : "border-border bg-muted"}`}>
                    <div className="text-muted-foreground mb-1 text-xs uppercase tracking-wider">{label}</div>
                    <div className={`font-mono text-lg ${highlight && label === "F1 Score" ? "text-amber-300 font-bold" : "text-foreground"}`}>
                      {value !== null ? value.toFixed(4) : "—"}
                    </div>
                  </div>
                ))}
              </div>

              {/* Confusion Matrix */}
              {result.confusionMatrix && result.confusionMatrix.length <= 6 && (
                <div>
                  <p className="text-xs uppercase tracking-wider text-muted-foreground mb-2">Confusion Matrix (Best Model)</p>
                  <div className="overflow-x-auto">
                    <table className="text-xs font-mono border-collapse">
                      <tbody>
                        {result.confusionMatrix.map((row, ri) => (
                          <tr key={ri}>
                            {row.map((cell, ci) => {
                              const isDiag = ri === ci;
                              return (
                                <td key={ci} className={`px-3 py-1.5 border border-border text-center min-w-[48px] ${isDiag ? "bg-primary/20 text-primary font-bold" : "bg-muted text-muted-foreground"}`}>
                                  {cell}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-1">Rows = actual, Columns = predicted. Diagonal = correct predictions.</p>
                </div>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
              {[
                { label: "RMSE", value: result.bestScore },
                { label: "MAE", value: result.mae ?? null },
              ].map(({ label, value }) => (
                <div key={label} className="p-3 rounded-md border border-border bg-muted text-center">
                  <div className="text-muted-foreground mb-1 text-xs uppercase tracking-wider">{label}</div>
                  <div className="font-mono text-lg text-foreground">
                    {value !== null ? value.toFixed(4) : "—"}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Comparison Chart */}
        <Card className="bg-card border-border shadow-md col-span-1">
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2 uppercase tracking-wider text-muted-foreground">
              <BarChart3 className="h-4 w-4" />
              Model Leaderboard
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={modelData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="hsl(var(--border))" />
                  <XAxis type="number" domain={['dataMin - 0.05', 'dataMax + 0.01']} stroke="hsl(var(--muted-foreground))" fontSize={12} tickFormatter={(val) => val.toFixed(2)} />
                  <YAxis dataKey="name" type="category" stroke="hsl(var(--muted-foreground))" fontSize={12} width={100} tick={{ fill: 'hsl(var(--foreground))' }} />
                  <RechartsTooltip 
                    cursor={{ fill: 'hsl(var(--muted))', opacity: 0.4 }}
                    contentStyle={{ backgroundColor: 'hsl(var(--popover))', borderColor: 'hsl(var(--border))', color: 'hsl(var(--popover-foreground))', borderRadius: '6px' }}
                    itemStyle={{ color: 'hsl(var(--primary))' }}
                  />
                  <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={24}>
                    {modelData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={index === 0 ? 'hsl(var(--primary))' : 'hsl(var(--muted-foreground))'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Feature Importance Chart */}
        <Card className="bg-card border-border shadow-md col-span-1">
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2 uppercase tracking-wider text-muted-foreground">
              <Hash className="h-4 w-4" />
              Feature Importance (Top 10)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={featureData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="hsl(var(--border))" />
                  <XAxis type="number" stroke="hsl(var(--muted-foreground))" fontSize={12} tickFormatter={(val) => `${val}%`} />
                  <YAxis dataKey="name" type="category" stroke="hsl(var(--muted-foreground))" fontSize={12} width={100} tick={{ fill: 'hsl(var(--foreground))' }} />
                  <RechartsTooltip 
                    cursor={{ fill: 'hsl(var(--muted))', opacity: 0.4 }}
                    formatter={(value: number) => [`${value}%`, 'Importance']}
                    contentStyle={{ backgroundColor: 'hsl(var(--popover))', borderColor: 'hsl(var(--border))', color: 'hsl(var(--popover-foreground))', borderRadius: '6px' }}
                  />
                  <Bar dataKey="value" fill="hsl(var(--chart-2))" radius={[0, 4, 4, 0]} barSize={20} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Dataset Details Table */}
      <Card className="bg-card border-border shadow-md">
        <CardHeader>
          <CardTitle className="text-sm uppercase tracking-wider text-muted-foreground">Dataset Profiling</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-6">
            <div className="p-3 bg-muted rounded-md border border-border text-center">
              <div className="text-muted-foreground mb-1 text-xs uppercase">Missing Values</div>
              <div className="font-mono text-lg text-foreground">{result.datasetSummary.missingValues}</div>
            </div>
            <div className="p-3 bg-muted rounded-md border border-border text-center">
              <div className="text-muted-foreground mb-1 text-xs uppercase">Duplicates Dropped</div>
              <div className="font-mono text-lg text-foreground">{result.datasetSummary.duplicatesRemoved}</div>
            </div>
            <div className="p-3 bg-muted rounded-md border border-border text-center">
              <div className="text-muted-foreground mb-1 text-xs uppercase">Numeric Cols</div>
              <div className="font-mono text-lg text-foreground">{result.datasetSummary.numericColumns.length}</div>
            </div>
            <div className="p-3 bg-muted rounded-md border border-border text-center">
              <div className="text-muted-foreground mb-1 text-xs uppercase">Categorical Cols</div>
              <div className="font-mono text-lg text-foreground">{result.datasetSummary.categoricalColumns.length}</div>
            </div>
          </div>
          
          <Separator className="bg-border my-4" />
          
          <div>
            <h4 className="text-xs uppercase tracking-wider text-muted-foreground mb-3">Schema Preview</h4>
            <div className="flex flex-wrap gap-2">
              {result.datasetSummary.columnNames.map((col) => {
                const isTarget = col === result.targetColumn;
                const isNumeric = result.datasetSummary.numericColumns.includes(col);
                return (
                  <Badge 
                    key={col} 
                    variant={isTarget ? "default" : "outline"}
                    className={`font-mono text-xs ${isTarget ? 'bg-primary text-primary-foreground' : 'border-border text-muted-foreground'} `}
                  >
                    {col}
                    {!isTarget && (
                      <span className="ml-1 opacity-50">
                        ({isNumeric ? 'num' : 'cat'})
                      </span>
                    )}
                  </Badge>
                );
              })}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Prediction / Explanation / Confidence panels */}
      <PredictionPanel jobId={jobId} />
    </div>
  );
}

function ResultsSkeleton() {
  return (
    <div className="flex flex-col gap-6 w-full animate-pulse">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Skeleton className="h-[120px] bg-card border border-border rounded-xl" />
        <Skeleton className="h-[120px] bg-card border border-border rounded-xl" />
        <Skeleton className="h-[120px] bg-card border border-border rounded-xl" />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Skeleton className="h-[400px] bg-card border border-border rounded-xl" />
        <Skeleton className="h-[400px] bg-card border border-border rounded-xl" />
      </div>
      <Skeleton className="h-[250px] bg-card border border-border rounded-xl" />
    </div>
  );
}
