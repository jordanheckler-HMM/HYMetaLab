import { z } from "zod";
export const PayloadSchema = z.object({
  experimentId: z.string(),
  simVersion: z.string(),
  runId: z.string(),
  seed: z.number().int(),
  horizonSteps: z.number().int(),
  agent_profile: z.object({
    calibration: z.number().min(0).max(1),
    coherence:   z.number().min(0).max(1),
    emergence:   z.number().min(0).max(1),
    noise:       z.number().min(0).max(1)
  }),
  policy_flags: z.object({
    interaction_style: z.enum(["telescope","oracle"]).optional(),
    network_mode: z.enum(["isolated","peer_feedback","mentor_hub"]).optional(),
    practice: z.enum(["none","attention_hygiene","prediction_journal","peer_review"]).optional()
  }).partial().optional(),
  environment: z.object({
    shock_severity: z.number().min(0).max(1),
    coordination_strength: z.number().min(0).max(1),
    goal_inequality: z.number().min(0).max(1),
    shock_profile: z.enum(["chronic_low","acute_high"]).optional()
  })
});
export type Payload = z.infer<typeof PayloadSchema>;
