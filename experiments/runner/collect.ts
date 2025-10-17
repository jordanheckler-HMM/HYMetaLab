import fs from 'fs-extra';
import path from 'path';
import { request } from 'undici';
import { ResultSchema } from '../schemas/results.zod';
import JsonDB from './jsondb';

const GATEWAY_DB = path.resolve(process.cwd(),'data','gateway','cci_experiments.json');
const RESULTS_DIR = path.resolve(process.cwd(),'data','experiments');

function initDb(){
  return new JsonDB(GATEWAY_DB);
}

export async function collectLoop(){
  const db = initDb();
  const rows = db.getPendingRuns();
  for (const r of rows as any[]){
    const runId = (r as any).run_id;
    let found = false;
    for (let port=5201; port<=5209; port++){
      try{
        const url = `http://localhost:${port}/run-status/${runId}`;
        const { body, statusCode } = await request(url,{method:'GET'});
        if (statusCode!==200) continue;
        const txt = await body.text();
        const res = JSON.parse(txt);
        const parsed = ResultSchema.parse(res);
        // write to results file and CSV
        db.insertResult(runId, parsed);
        const expDir = path.join(RESULTS_DIR, parsed.runId.split('_')[0]);
        await fs.ensureDir(expDir);
        const csvPath = path.join(expDir,'results.csv');
        const prev = fs.existsSync(csvPath) ? await fs.readFile(csvPath,'utf8') : 'runId,simInstanceId,startedAt,finishedAt,ok,survival_rate,collapse_risk,aggression_delta,collective_cci_delta,shock_tolerance,branch_selected\n';
        const line = `${parsed.runId},${parsed.simInstanceId},${parsed.startedAt},${parsed.finishedAt},${parsed.ok},${parsed.survival_rate},${parsed.collapse_risk},${parsed.aggression_delta},${parsed.collective_cci_delta},${parsed.shock_tolerance},${parsed.branch_selected || ''}\n`;
        await fs.writeFile(csvPath, prev + line, 'utf8');
        // update run status
        db.upsertRun(runId, { ...(r as any), status: 'done', finished_at: new Date().toISOString() });
        found = true;
        console.log(`Collected result for ${runId} from port ${port}`);
        break;
      }catch(e){
        // ignore and continue
      }
    }
    if (!found) console.log(`No result yet for ${runId}`);
  }
}

if (require.main===module) collectLoop().catch(e=>{console.error(e);process.exit(1)});
