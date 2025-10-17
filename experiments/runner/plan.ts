import fs from 'fs-extra';
import YAML from 'yaml';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';

const CONFIG = path.resolve(__dirname, '..', 'config', 'experiments_quickpass.yaml');
const RESULTS_DIR = path.resolve(process.cwd(), 'data', 'experiments');

function cartesian<T>(obj: Record<string, T[]>){
  const keys = Object.keys(obj);
  const res: any[] = [{}];
  for (const k of keys){
    const arr = obj[k];
    const next: any[] = [];
    for (const r of res) for (const v of arr) next.push({...r,[k]:v});
    res.splice(0,res.length,...next);
  }
  return res;
}

export async function planAll(){
  const txt = await fs.readFile(CONFIG,'utf8');
  const doc = YAML.parse(txt);
  await fs.ensureDir(RESULTS_DIR);
  for (const exp of doc.experiments){
    const factors = exp.factors || {};
    const grid = cartesian(factors);
    const expDir = path.join(RESULTS_DIR, exp.id);
    await fs.ensureDir(expDir);
    const plan:any[] = [];
    let cellIndex = 0;
    for (const cell of grid){
      const seeds = doc.defaults.ensemble_seeds || 2;
      for (let s=0;s<seeds;s++){
        const runId = `${exp.id}_${cellIndex}_${s}`;
        const payload = { experimentId: exp.id, simVersion: doc.sim_version, runId, seed: s, horizonSteps: doc.defaults.horizon_steps, agent_profile: { calibration: cell.base_calibration ?? 0.7, coherence: cell.base_coherence ?? 0.7, emergence: cell.base_emergence ?? 0.7, noise: cell.base_noise ?? 0.2 }, policy_flags: {}, environment: { shock_severity: cell.shock_severity ?? 0.3, coordination_strength: cell.coordination_strength ?? 0.5, goal_inequality: cell.goal_inequality ?? 0.4 } };
        plan.push({cellIndex, runId, payload});
      }
      cellIndex++;
    }
    await fs.writeJson(path.join(expDir,'run_plan.json'),plan,{spaces:2});
    console.log(`Wrote plan for ${exp.id}: ${plan.length} runs to ${expDir}`);
  }
}

if (require.main === module) planAll().catch(e=>{console.error(e);process.exit(1)});
