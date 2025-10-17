import fs from 'fs-extra';
import path from 'path';
import { mean, std } from 'mathjs';
import YAML from 'yaml';

const RESULTS_DIR = path.resolve(process.cwd(),'data','experiments');
const CONFIG = path.resolve(__dirname,'..','config','experiments_quickpass.yaml');

function ci95(arr:number[]){
  if (arr.length===0) return {mean:NaN, ci95:NaN};
  const m = mean(arr as any) as number;
  const s = std(arr as any) as number;
  const se = s / Math.sqrt(arr.length);
  const ci = 1.96 * se;
  return {mean: m, ci95: ci};
}

export async function summarizeAll(){
  const cfgTxt = await fs.readFile(CONFIG,'utf8');
  const cfg = YAML.parse(cfgTxt);
  const allowedIds = new Set((cfg.experiments || []).map((e:any)=> e.id));
  const argv = process.argv.slice(2);
  const idIndex = argv.findIndex(a=>a==='--id');
  const filterId = idIndex>=0 ? argv[idIndex+1] : null;

  const expDirs = await fs.readdir(RESULTS_DIR);
  for (const expId of expDirs){
    if (!allowedIds.has(expId)) continue;
    if (filterId && filterId !== expId) continue;
    const csvPath = path.join(RESULTS_DIR, expId, 'results.csv');
    if (!fs.existsSync(csvPath)) continue;
    const txt = await fs.readFile(csvPath,'utf8');
    const rows = txt.trim().split('\n').slice(1).map(l=>l.split(','));
    const survival = rows.map(r=>Number(r[5]));
    const collapse = rows.map(r=>Number(r[6]));
    const shock_tol = rows.map(r=>Number(r[9]));
    const cci = rows.map(r=>Number(r[8]));
    const agg = rows.map(r=>Number(r[7]));
    const s1 = ci95(survival);
    const s2 = ci95(collapse);
    const s3 = ci95(shock_tol);
    const s4 = ci95(cci);
    const s5 = ci95(agg);
    const md = `# Summary for ${expId}\n\nN runs: ${rows.length}\n\n- survival_rate: mean=${s1.mean.toFixed(3)} ± ${s1.ci95.toFixed(3)}\n- collapse_risk: mean=${s2.mean.toFixed(3)} ± ${s2.ci95.toFixed(3)}\n- shock_tolerance: mean=${s3.mean.toFixed(3)} ± ${s3.ci95.toFixed(3)}\n- collective_cci_delta: mean=${s4.mean.toFixed(3)} ± ${s4.ci95.toFixed(3)}\n- aggression_delta: mean=${s5.mean.toFixed(3)} ± ${s5.ci95.toFixed(3)}\n`;
    await fs.writeFile(path.join(RESULTS_DIR, expId, 'summary.md'), md, 'utf8');
    console.log(`Wrote summary for ${expId}`);
  }
}

if (require.main===module) summarizeAll().catch(e=>{console.error(e);process.exit(1)});
