import { z } from "zod";
export const ResultSchema = z.object({
  runId: z.string(),
  simInstanceId: z.string(),
  startedAt: z.string(),
  finishedAt: z.string(),
  ok: z.boolean(),
  survival_rate: z.number().min(0).max(1),
  collapse_risk: z.number().min(0).max(1),
  aggression_delta: z.number(),
  collective_cci_delta: z.number(),
  shock_tolerance: z.number().min(0).max(1),
  branch_selected: z.enum(["meaning","training"]).optional(),
  logs_url: z.string().optional()
});
export type SimResult = z.infer<typeof ResultSchema>;
