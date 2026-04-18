# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)
- **Python**: 3.11 with scikit-learn, pandas, kaggle, joblib, flask

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

## Artifacts

### AutoML Studio (`artifacts/automl-studio`)
- React + Vite frontend at `/`
- Dark, technical UI with recharts for visualizations
- Kaggle dataset input with autocomplete search
- Real-time pipeline progress tracking (polls every 2s)
- Feature importance and model comparison charts

### API Server (`artifacts/api-server`)
- Express 5 + TypeScript
- Routes in `src/routes/pipeline.ts` — proxies to Python ML service
- Spawns Python Flask ML service on port 5001 at startup

### Python ML Service (`artifacts/api-server/src/lib/ml-pipeline.py`)
- Flask microservice on port 5001
- Downloads Kaggle datasets via Kaggle CLI
- Cleans data: deduplication, missing value imputation, categorical encoding
- Auto-detects target column and problem type (classification/regression)
- Trains: Random Forest, Gradient Boosting, Logistic/Linear Regression, Decision Tree
- Selects best model by accuracy (classification) or RMSE (regression)
- Saves best model to `/tmp/automl-studio/models/` with joblib
- Returns feature importance from tree-based models

## Kaggle Setup

Users need Kaggle credentials to download datasets:
1. Set `KAGGLE_USERNAME` and `KAGGLE_KEY` as environment variables, OR
2. Enter credentials in the UI's "Kaggle Credentials" collapsible section

## Architecture

```
Browser → React UI → /api → Express → Python Flask (port 5001)
                                        ↓
                              Kaggle CLI download
                                        ↓
                              pandas cleaning
                                        ↓
                              sklearn model training
                                        ↓
                              joblib model save
```

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.
