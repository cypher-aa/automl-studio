import { useState, useRef } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Play, Database, KeyRound, User, ChevronDown, ChevronUp, Search, Loader2 } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { useRunPipeline, useSearchKaggleDatasets, getSearchKaggleDatasetsQueryKey } from "@workspace/api-client-react";
import { useDebounce } from "@/hooks/use-debounce";
import { useToast } from "@/hooks/use-toast";
import { useQueryClient } from "@tanstack/react-query";

const formSchema = z.object({
  datasetName: z.string().min(1, "Dataset name is required"),
  targetColumn: z.string().optional(),
  kaggleUsername: z.string().optional(),
  kaggleKey: z.string().optional(),
});

type FormValues = z.infer<typeof formSchema>;

export function PipelineForm({ onJobStarted }: { onJobStarted: (jobId: string) => void }) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [credentialsOpen, setCredentialsOpen] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      datasetName: "",
      targetColumn: "",
      kaggleUsername: "",
      kaggleKey: "",
    },
  });

  const datasetSearchTerm = form.watch("datasetName");
  const debouncedSearchTerm = useDebounce(datasetSearchTerm, 300);

  const { data: searchResults, isFetching: isSearching } = useSearchKaggleDatasets(
    { query: debouncedSearchTerm },
    {
      query: {
        enabled: debouncedSearchTerm.length > 2,
        queryKey: getSearchKaggleDatasetsQueryKey({ query: debouncedSearchTerm }),
      },
    }
  );

  const runPipeline = useRunPipeline();

  const onSubmit = (values: FormValues) => {
    runPipeline.mutate(
      {
        data: {
          datasetName: values.datasetName,
          targetColumn: values.targetColumn || null,
          kaggleUsername: values.kaggleUsername || null,
          kaggleKey: values.kaggleKey || null,
        },
      },
      {
        onSuccess: (data) => {
          toast({
            title: "Pipeline Started",
            description: `Job ${data.jobId} initialized for ${data.datasetName}`,
          });
          onJobStarted(data.jobId);
          // Invalidate jobs list
          queryClient.invalidateQueries({ queryKey: ["/api/pipeline/jobs"] });
        },
        onError: (error: any) => {
          toast({
            title: "Failed to start pipeline",
            description: error?.error || "Unknown error occurred",
            variant: "destructive",
          });
        },
      }
    );
  };

  return (
    <Card className="border-border bg-card shadow-md">
      <CardHeader className="pb-4">
        <CardTitle className="text-lg flex items-center gap-2">
          <Database size={18} className="text-primary" />
          New Pipeline
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4" data-testid="form-pipeline">
            <FormField
              control={form.control}
              name="datasetName"
              render={({ field }) => (
                <FormItem className="relative" ref={wrapperRef}>
                  <FormLabel>Kaggle Dataset</FormLabel>
                  <FormControl>
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="e.g. titanic or username/dataset"
                        className="pl-9 font-mono"
                        {...field}
                        onChange={(e) => {
                          field.onChange(e);
                          setShowSuggestions(true);
                        }}
                        onFocus={() => setShowSuggestions(true)}
                        onBlur={() => {
                          // Small delay to allow click on suggestion
                          setTimeout(() => setShowSuggestions(false), 200);
                        }}
                        data-testid="input-dataset-name"
                      />
                    </div>
                  </FormControl>
                  <FormMessage />

                  {/* Autocomplete Dropdown */}
                  {showSuggestions && debouncedSearchTerm.length > 2 && (
                    <div className="absolute z-50 w-full mt-1 bg-popover border border-border rounded-md shadow-lg overflow-hidden">
                      {isSearching ? (
                        <div className="p-3 text-sm text-muted-foreground flex items-center gap-2">
                          <Loader2 className="h-4 w-4 animate-spin" /> Searching...
                        </div>
                      ) : searchResults && searchResults.length > 0 ? (
                        <ul className="max-h-[250px] overflow-auto py-1">
                          {searchResults.map((dataset) => (
                            <li
                              key={dataset.ref}
                              className="px-3 py-2 hover:bg-muted cursor-pointer text-sm font-sans flex flex-col"
                              onMouseDown={(e) => {
                                e.preventDefault(); // Prevent blur
                                form.setValue("datasetName", dataset.ref);
                                setShowSuggestions(false);
                              }}
                            >
                              <span className="font-medium text-foreground">{dataset.title}</span>
                              <span className="text-xs text-muted-foreground font-mono">{dataset.ref} • {dataset.size}</span>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <div className="p-3 text-sm text-muted-foreground">No datasets found</div>
                      )}
                    </div>
                  )}
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="targetColumn"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Target Column (Optional)</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="Auto-detect if blank"
                      className="font-mono"
                      {...field}
                      data-testid="input-target-column"
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <Collapsible open={credentialsOpen} onOpenChange={setCredentialsOpen} className="border border-border rounded-md">
              <CollapsibleTrigger asChild>
                <Button
                  variant="ghost"
                  className="w-full flex justify-between items-center p-3 h-auto font-sans text-sm rounded-none"
                  type="button"
                >
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <KeyRound size={16} />
                    <span>Kaggle Credentials (Optional)</span>
                  </div>
                  {credentialsOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="p-3 pt-0 space-y-3 bg-muted/20 border-t border-border">
                <FormField
                  control={form.control}
                  name="kaggleUsername"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="text-xs text-muted-foreground">Username</FormLabel>
                      <FormControl>
                        <div className="relative">
                          <User className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                          <Input className="pl-9 h-8 text-xs font-mono" placeholder="username" {...field} data-testid="input-kaggle-username" />
                        </div>
                      </FormControl>
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="kaggleKey"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="text-xs text-muted-foreground">API Key</FormLabel>
                      <FormControl>
                        <div className="relative">
                          <KeyRound className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                          <Input
                            type="password"
                            className="pl-9 h-8 text-xs font-mono"
                            placeholder="••••••••••••••••"
                            {...field}
                            data-testid="input-kaggle-key"
                          />
                        </div>
                      </FormControl>
                    </FormItem>
                  )}
                />
              </CollapsibleContent>
            </Collapsible>

            <Button
              type="submit"
              className="w-full font-bold uppercase tracking-wider mt-4"
              disabled={runPipeline.isPending}
              data-testid="button-run-pipeline"
            >
              {runPipeline.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Initiating...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" fill="currentColor" /> Run Pipeline
                </>
              )}
            </Button>
          </form>
        </Form>
      </CardContent>
    </Card>
  );
}
