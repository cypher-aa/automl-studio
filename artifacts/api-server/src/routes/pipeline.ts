import { Router, type IRouter } from "express";
import {
  RunPipelineBody,
  GetPipelineStatusParams,
  GetPipelineResultParams,
  SearchKaggleDatasetsQueryParams,
} from "@workspace/api-zod";
import { logger } from "../lib/logger";

const router: IRouter = Router();

const ML_SERVICE_URL = `http://localhost:${process.env.ML_SERVICE_PORT || "5001"}`;

async function proxyToML(path: string, method: string = "GET", body?: unknown): Promise<Response> {
  const options: RequestInit = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body) {
    options.body = JSON.stringify(body);
  }
  return fetch(`${ML_SERVICE_URL}${path}`, options);
}

router.post("/pipeline/run", async (req, res): Promise<void> => {
  const parsed = RunPipelineBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.message });
    return;
  }

  try {
    const mlRes = await proxyToML("/pipeline/run", "POST", parsed.data);
    const data = await mlRes.json();
    res.status(mlRes.status).json(data);
  } catch (err) {
    req.log.error({ err }, "ML service error on /pipeline/run");
    res.status(500).json({ error: "ML service is not available. Please try again in a moment." });
  }
});

router.get("/pipeline/status/:jobId", async (req, res): Promise<void> => {
  const rawId = Array.isArray(req.params.jobId) ? req.params.jobId[0] : req.params.jobId;
  const parsed = GetPipelineStatusParams.safeParse({ jobId: rawId });
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.message });
    return;
  }

  try {
    const mlRes = await proxyToML(`/pipeline/status/${parsed.data.jobId}`);
    const data = await mlRes.json();
    res.status(mlRes.status).json(data);
  } catch (err) {
    req.log.error({ err }, "ML service error on /pipeline/status");
    res.status(500).json({ error: "ML service is not available." });
  }
});

router.get("/pipeline/result/:jobId", async (req, res): Promise<void> => {
  const rawId = Array.isArray(req.params.jobId) ? req.params.jobId[0] : req.params.jobId;
  const parsed = GetPipelineResultParams.safeParse({ jobId: rawId });
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.message });
    return;
  }

  try {
    const mlRes = await proxyToML(`/pipeline/result/${parsed.data.jobId}`);
    const data = await mlRes.json();
    res.status(mlRes.status).json(data);
  } catch (err) {
    req.log.error({ err }, "ML service error on /pipeline/result");
    res.status(500).json({ error: "ML service is not available." });
  }
});

router.get("/pipeline/jobs", async (_req, res): Promise<void> => {
  try {
    const mlRes = await proxyToML("/pipeline/jobs");
    const data = await mlRes.json();
    res.status(mlRes.status).json(data);
  } catch (err) {
    logger.error({ err }, "ML service error on /pipeline/jobs");
    res.status(500).json({ error: "ML service is not available." });
  }
});

router.get("/kaggle/search", async (req, res): Promise<void> => {
  const parsed = SearchKaggleDatasetsQueryParams.safeParse(req.query);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.message });
    return;
  }

  try {
    const mlRes = await proxyToML(`/kaggle/search?query=${encodeURIComponent(parsed.data.query)}`);
    const data = await mlRes.json();
    res.status(mlRes.status).json(data);
  } catch (err) {
    req.log.error({ err }, "ML service error on /kaggle/search");
    res.status(500).json({ error: "ML service is not available." });
  }
});

export default router;
