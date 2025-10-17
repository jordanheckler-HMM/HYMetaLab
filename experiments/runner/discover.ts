import { request } from 'undici';
import fs from 'fs-extra';

const PORTS = [5201,5202,5203,5204,5205,5206,5207,5208,5209];

export type Instance = {
  simInstanceId: string;
  port: number;
  metricsPort?: number;
  dataDir: string;
  dbPath: string;
  instanceFingerprint: string;
  rngSeed: number;
}

export async function discoverInstances(): Promise<Instance[]> {
  const found: Instance[] = [];
  for (const port of PORTS) {
    const url = `http://localhost:${port}/healthz`;
    try {
      const { body, statusCode } = await request(url, { method: 'GET' });
      if (statusCode !== 200) continue;
      const text = await body.text();
      const json = JSON.parse(text);
      found.push({
        simInstanceId: json.simInstanceId,
        port: port,
        metricsPort: json.metricsPort,
        dataDir: json.dataDir,
        dbPath: json.dbPath,
        instanceFingerprint: json.instanceFingerprint,
        rngSeed: json.rngSeed
      });
    } catch (err) {
      // ignore unreachable ports
    }
  }
  if (found.length === 0) throw new Error('No instances discovered on the configured ports.');
  // print concise table
  console.table(found.map(f => ({id: f.simInstanceId, port: f.port, db: f.dbPath})));
  return found;
}

if (require.main === module) {
  (async ()=>{
    try{
      await discoverInstances();
    }catch(e){
      console.error(e);
      process.exit(1);
    }
  })();
}
