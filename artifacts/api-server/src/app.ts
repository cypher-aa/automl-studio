import express, { type Express } from "express";
import cors from "cors";
import pinoHttp from "pino-http";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import router from "./routes";
import { logger } from "./lib/logger";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app: Express = express();

app.use(
  pinoHttp({
    logger,
    serializers: {
      req(req) {
        return {
          id: req.id,
          method: req.method,
          url: req.url?.split("?")[0],
        };
      },
      res(res) {
        return {
          statusCode: res.statusCode,
        };
      },
    },
  }),
);
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use("/api", router);

// Start Python ML service
const mlPort = process.env.ML_SERVICE_PORT || "5001";
// In development, the Python script is in src/lib/. After build, it's copied to dist/
const mlScriptPath = process.env.NODE_ENV === "development"
  ? path.resolve(__dirname, "../src/lib/ml-pipeline.py")
  : path.resolve(__dirname, "ml-pipeline.py");

function startMLService() {
  const mlEnv = {
    ...process.env,
    ML_SERVICE_PORT: mlPort,
  };

  const mlProcess = spawn("python3", [mlScriptPath], {
    env: mlEnv,
    stdio: "pipe",
  });

  mlProcess.stdout?.on("data", (data: Buffer) => {
    logger.info({ service: "ml-pipeline" }, data.toString().trim());
  });

  mlProcess.stderr?.on("data", (data: Buffer) => {
    const msg = data.toString().trim();
    if (msg) {
      logger.info({ service: "ml-pipeline" }, msg);
    }
  });

  mlProcess.on("close", (code: number) => {
    logger.warn({ code, service: "ml-pipeline" }, "ML service exited, restarting in 3s...");
    setTimeout(startMLService, 3000);
  });

  mlProcess.on("error", (err: Error) => {
    logger.error({ err, service: "ml-pipeline" }, "Failed to start ML service");
    setTimeout(startMLService, 5000);
  });

  logger.info({ port: mlPort }, "ML pipeline service starting");
}

startMLService();

export default app;
