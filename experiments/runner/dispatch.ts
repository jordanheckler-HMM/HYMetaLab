import fs from 'fs-extra';
import path from 'path';
import YAML from 'yaml';
import { request } from 'undici';
import { discoverInstances } from './discover';
import JsonDB from './jsondb';

const CONFIG = path.resolve(__dirname, '..', 'config', 'experiments_quickpass.yaml');
const GATEWAY_DB = path.resolve(process.cwd(),'data','gateway','cci_experiments.json');
const RESULTS_DIR = path.resolve(process.cwd(),'data','experiments');
const MAX_WALL_TIME_MS = 90 * 60 * 1000; // 90 minutes
const HEARTBEAT_INTERVAL_MS = 2 * 60 * 1000; // 2 minutes
const MAX_CONCURRENCY_PER_INSTANCE = 3;

function sleep(ms:number){ return new Promise(res=>setTimeout(res,ms)); }

function initDb(){
  fs.ensureDirSync(path.dirname(GATEWAY_DB));
  return new JsonDB(GATEWAY_DB);
}

export async function dispatchAll(){
  const startTime = Date.now();
  const instances = await discoverInstances();
  const db = initDb();
  // simple round robin
  const instanceStates: Record<string, number> = {};
  for (const inst of instances) instanceStates[inst.simInstanceId]=0;
  
  let totalDispatched = 0;
  let lastHeartbeat = Date.now();

  // determine allowed experiment ids from quickpass config
  const cfgTxt = await fs.readFile(CONFIG,'utf8');
  const cfg = YAML.parse(cfgTxt);
  const allowedIds = new Set((cfg.experiments || []).map((e:any)=> e.id));
  const argv = process.argv.slice(2);
  const idIndex = argv.findIndex(a=>a==='--id');
  const filterId = idIndex>=0 ? argv[idIndex+1] : null;
  const expDirs = await fs.readdir(RESULTS_DIR);
  for (const expId of expDirs){
    if (!allowedIds.has(expId)) { continue; }
    if (filterId && filterId !== expId) { continue; }
    // Check time cap
    if (Date.now() - startTime > MAX_WALL_TIME_MS) {
      console.log(`⏰ TIME CAP: Stopping dispatch after ${((Date.now() - startTime)/60000).toFixed(1)} minutes. Use --resume to continue.`);
      break;
    }
    
    const planPath = path.join(RESULTS_DIR, expId, 'run_plan.json');
    if (!fs.existsSync(planPath)) continue;
    const plan = await fs.readJson(planPath);
    for (const run of plan){
      // Heartbeat progress
      if (Date.now() - lastHeartbeat > HEARTBEAT_INTERVAL_MS) {
        console.log(`💓 HEARTBEAT: ${totalDispatched} runs dispatched, ${((Date.now() - startTime)/60000).toFixed(1)}min elapsed`);
        lastHeartbeat = Date.now();
      }
      // enforce per-instance concurrency limit
      let pick: any = null;
      // try to find an instance with capacity
      for (const inst of instances){
        if ((instanceStates[inst.simInstanceId] || 0) < MAX_CONCURRENCY_PER_INSTANCE){
          if (pick === null || instanceStates[inst.simInstanceId] < instanceStates[pick.simInstanceId]) pick = inst;
        }
      }
      // if none available, wait briefly then retry (respecting wall time)
      let waitLoops = 0;
      while (!pick){
        if (Date.now() - startTime > MAX_WALL_TIME_MS) { console.log('Time cap reached while waiting for capacity'); break; }
        await sleep(300);
        waitLoops++;
        for (const inst of instances){
          if ((instanceStates[inst.simInstanceId] || 0) < MAX_CONCURRENCY_PER_INSTANCE){
            pick = inst; break;
          }
        }
        if (waitLoops > 200) { console.log('No capacity after waiting, skipping run', run.runId); break; }
      }
      if (!pick) continue;
      const payload = run.payload;
      // post
      const url = `http://localhost:${pick.port}/enqueue-run`;
      try{
        await request(url,{method:'POST', body: JSON.stringify(payload), headers:{'content-type':'application/json'}});
        instanceStates[pick.simInstanceId]++;
        totalDispatched++;
        db.upsertRun(run.runId, { run_id: run.runId, experiment_id: expId, instance_id: pick.simInstanceId, seed: payload.seed, payload_json: payload, status: 'pending', sim_version: payload.simVersion });
        console.log(`Dispatched ${run.runId} -> ${pick.simInstanceId}`);
      }catch(e){
        console.error('Dispatch error',e);
        db.upsertRun(run.runId, { run_id: run.runId, experiment_id: expId, instance_id: pick.simInstanceId, seed: payload.seed, payload_json: payload, status: 'error', sim_version: payload.simVersion, error_msg: String(e) });
      }
    }
  }
}

if (require.main===module) dispatchAll().catch(e=>{console.error(e);process.exit(1)});
