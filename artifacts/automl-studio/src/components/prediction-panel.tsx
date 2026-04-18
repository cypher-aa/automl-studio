import { useState } from "react";
import { usePredictPipeline, useGetPipelineResult, getGetPipelineResultQueryKey } from "@workspace/api-client-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { AlertCircle, Brain, BarChart2, Sparkles, ShieldCheck, ShieldAlert, Shield, ChevronDown, ChevronUp } from "lucide-react";

interface PredictionPanelProps {
  jobId: string;
}

const CONFIDENCE_CONFIG = {
  High: {
    color: "text-emerald-400",
    bg: "bg-emerald-500/10 border-emerald-500/30",
    icon: ShieldCheck,
    label: "High Confidence",
  },
  Medium: {
    color: "text-amber-400",
    bg: "bg-amber-500/10 border-amber-500/30",
    icon: Shield,
    label: "Medium Confidence",
  },
  Low: {
    color: "text-red-400",
    bg: "bg-red-500/10 border-red-500/30",
    icon: ShieldAlert,
    label: "Low Confidence",
  },
} as const;

export function PredictionPanel({ jobId }: PredictionPanelProps) {
  const [formValues, setFormValues] = useState<Record<string, string>>({});
  const [predictionResult, setPredictionResult] = useState<{
    predictedValue: string;
    topFeatures: { feature: string; importance: number }[];
    explanation: string;
    confidence: string | null;
    confidenceDetail: string | null;
    problemType: string;
  } | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showAllFields, setShowAllFields] = useState(false);

  const { data: result } = useGetPipelineResult(jobId, {
    query: {
      enabled: !!jobId,
      queryKey: getGetPipelineResultQueryKey(jobId),
    },
  });

  const predictMutation = usePredictPipeline();

  if (!result) return null;

  const featureNames = result.featureNames ?? [];
  if (featureNames.length === 0) return null;

  const INITIAL_SHOW = 8;
  const visibleFeatures = showAllFields ? featureNames : featureNames.slice(0, INITIAL_SHOW);
  const hasMore = featureNames.length > INITIAL_SHOW;

  // Compute max importance for bar scaling
  const maxImportance = predictionResult?.topFeatures.length
    ? Math.max(...predictionResult.topFeatures.map((f) => f.importance))
    : 1;

  function handleChange(feature: string, value: string) {
    setFormValues((prev) => ({ ...prev, [feature]: value }));
  }

  async function handlePredict() {
    setErrorMsg(null);
    setPredictionResult(null);

    // Validate: all visible fields must have a value
    const missing = featureNames.filter((f) => !formValues[f]?.trim());
    if (missing.length > 0 && featureNames.length <= INITIAL_SHOW) {
      setErrorMsg(`Please fill in all feature values before predicting.`);
      return;
    }

    try {
      const res = await predictMutation.mutateAsync({
        jobId,
        data: { features: formValues },
      });
      setPredictionResult(res);
    } catch (err: unknown) {
      const message =
        err instanceof Error
          ? err.message
          : "Prediction failed. Please check your inputs and try again.";
      setErrorMsg(message);
    }
  }

  const confidence = predictionResult?.confidence as keyof typeof CONFIDENCE_CONFIG | null;
  const confConfig = confidence ? CONFIDENCE_CONFIG[confidence] ?? null : null;
  const ConfIcon = confConfig?.icon ?? ShieldCheck;

  return (
    <div className="flex flex-col gap-6 mt-2 animate-in fade-in slide-in-from-bottom-4 duration-700 fill-mode-both">
      <Separator className="bg-border" />

      {/* ── Section 1: Make a Prediction ── */}
      <Card className="bg-card border-border shadow-md">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2 uppercase tracking-wider text-muted-foreground">
            <Sparkles className="h-4 w-4 text-primary" />
            Make a Prediction
          </CardTitle>
          <p className="text-xs text-muted-foreground font-sans mt-1">
            Enter feature values to get a prediction from the best trained model ({result.bestModel}).
          </p>
        </CardHeader>

        <CardContent className="flex flex-col gap-4">
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            {visibleFeatures.map((feature) => (
              <div key={feature} className="flex flex-col gap-1">
                <Label
                  htmlFor={`feature-${feature}`}
                  className="text-xs text-muted-foreground font-mono truncate"
                  title={feature}
                >
                  {feature}
                </Label>
                <Input
                  id={`feature-${feature}`}
                  data-testid={`input-feature-${feature}`}
                  placeholder="0"
                  value={formValues[feature] ?? ""}
                  onChange={(e) => handleChange(feature, e.target.value)}
                  className="h-8 text-xs font-mono bg-muted border-border focus:border-primary"
                />
              </div>
            ))}
          </div>

          {hasMore && (
            <button
              type="button"
              onClick={() => setShowAllFields((v) => !v)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-primary transition-colors"
              data-testid="button-toggle-fields"
            >
              {showAllFields ? (
                <><ChevronUp className="h-3 w-3" /> Show fewer fields</>
              ) : (
                <><ChevronDown className="h-3 w-3" /> Show all {featureNames.length} fields</>
              )}
            </button>
          )}

          {errorMsg && (
            <div className="flex items-start gap-2 rounded-md bg-destructive/10 border border-destructive/30 px-3 py-2 text-xs text-destructive font-sans">
              <AlertCircle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          <Button
            data-testid="button-predict"
            onClick={handlePredict}
            disabled={predictMutation.isPending}
            className="w-full sm:w-auto font-mono uppercase tracking-widest text-xs"
          >
            {predictMutation.isPending ? "Predicting..." : "Predict"}
          </Button>

          {/* Prediction output */}
          {predictionResult && (
            <div
              className="rounded-md border border-primary/40 bg-primary/5 px-4 py-3 flex items-center gap-3 animate-in fade-in duration-300"
              data-testid="result-prediction"
            >
              <div className="h-8 w-8 rounded bg-primary/20 flex items-center justify-center text-primary shrink-0">
                <Sparkles className="h-4 w-4" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider mb-0.5">
                  Predicted {result.targetColumn}
                </p>
                <p className="text-xl font-bold font-mono text-foreground">
                  {predictionResult.predictedValue}
                </p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* ── Section 2 & 3: Explanation + Confidence (only after a prediction) ── */}
      {predictionResult && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Explanation */}
          <Card className="bg-card border-border shadow-md">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2 uppercase tracking-wider text-muted-foreground">
                <Brain className="h-4 w-4 text-primary" />
                Why this prediction?
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              {/* Natural language explanation */}
              <p
                className="text-sm font-sans text-foreground leading-relaxed bg-muted/50 border border-border rounded-md px-3 py-2"
                data-testid="text-explanation"
              >
                {predictionResult.explanation}
              </p>

              {/* Top features */}
              {predictionResult.topFeatures.length > 0 && (
                <div className="flex flex-col gap-2">
                  <p className="text-xs uppercase tracking-wider text-muted-foreground flex items-center gap-1">
                    <BarChart2 className="h-3.5 w-3.5" />
                    Top influencing features
                  </p>
                  {predictionResult.topFeatures.map((f, i) => {
                    const barWidth = maxImportance > 0
                      ? Math.round((f.importance / maxImportance) * 100)
                      : 0;
                    return (
                      <div key={f.feature} className="flex flex-col gap-1">
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-mono text-foreground flex items-center gap-1.5">
                            <span className="text-muted-foreground w-4 text-right">{i + 1}.</span>
                            {f.feature}
                          </span>
                          <span className="font-mono text-muted-foreground">
                            {(f.importance * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                          <div
                            className="h-full rounded-full bg-primary transition-all duration-500"
                            style={{ width: `${barWidth}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {predictionResult.topFeatures.length === 0 && (
                <p className="text-xs text-muted-foreground font-sans">
                  Feature importance is not available for this model type.
                </p>
              )}
            </CardContent>
          </Card>

          {/* Confidence */}
          <Card className="bg-card border-border shadow-md">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2 uppercase tracking-wider text-muted-foreground">
                <ShieldCheck className="h-4 w-4 text-primary" />
                Confidence
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              {confConfig ? (
                <div
                  className={`flex items-center gap-3 rounded-md border px-4 py-3 ${confConfig.bg}`}
                  data-testid="result-confidence"
                >
                  <ConfIcon className={`h-6 w-6 shrink-0 ${confConfig.color}`} />
                  <div>
                    <p className={`text-base font-bold font-mono ${confConfig.color}`}>
                      {confConfig.label}
                    </p>
                    {predictionResult.confidenceDetail && (
                      <p className="text-xs text-muted-foreground font-mono mt-0.5">
                        {predictionResult.confidenceDetail}
                      </p>
                    )}
                  </div>
                </div>
              ) : (
                <div className="rounded-md border border-border bg-muted/40 px-4 py-3">
                  <p className="text-sm text-muted-foreground font-sans">
                    Confidence level not available.
                  </p>
                </div>
              )}

              {/* Context card */}
              <div className="rounded-md bg-muted/30 border border-border px-3 py-3 flex flex-col gap-2 text-xs font-sans">
                <p className="text-muted-foreground uppercase tracking-wider font-mono text-[10px]">
                  How confidence is determined
                </p>
                {predictionResult.problemType === "classification" ? (
                  <ul className="flex flex-col gap-1 text-muted-foreground">
                    <li className="flex items-center gap-2">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
                      Accuracy &ge; 90% → High
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="h-1.5 w-1.5 rounded-full bg-amber-400 shrink-0" />
                      Accuracy &ge; 75% → Medium
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="h-1.5 w-1.5 rounded-full bg-red-400 shrink-0" />
                      Below 75% → Low
                    </li>
                  </ul>
                ) : (
                  <ul className="flex flex-col gap-1 text-muted-foreground">
                    <li className="flex items-center gap-2">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
                      RMSE &lt; 30,000 → High
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="h-1.5 w-1.5 rounded-full bg-amber-400 shrink-0" />
                      RMSE &lt; 70,000 → Medium
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="h-1.5 w-1.5 rounded-full bg-red-400 shrink-0" />
                      RMSE &ge; 70,000 → Low
                    </li>
                  </ul>
                )}
                <div className="mt-2 flex flex-wrap gap-2">
                  <Badge variant="outline" className="font-mono text-[10px] border-border text-muted-foreground">
                    Model: {result.bestModel}
                  </Badge>
                  <Badge variant="outline" className="font-mono text-[10px] border-border text-muted-foreground">
                    {result.scoreMetric.toUpperCase()}: {result.bestScore.toFixed(4)}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
