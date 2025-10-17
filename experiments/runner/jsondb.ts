import fs from 'fs-extra';
import path from 'path';

export class JsonDB {
  file: string;
  data: any;
  constructor(filePath: string){
    this.file = filePath;
    fs.ensureFileSync(this.file);
    try{
      this.data = fs.readJsonSync(this.file);
    }catch(e){ this.data = { runs: {}, results: {} }; fs.writeJsonSync(this.file,this.data,{spaces:2}); }
  }
  save(){ fs.writeJsonSync(this.file, this.data, {spaces:2}); }
  upsertRun(runId: string, obj: any){ this.data.runs[runId] = {...(this.data.runs[runId]||{}), ...obj}; this.save(); }
  getPendingRuns(): any[]{ return Object.values(this.data.runs).filter((r:any)=> r.status==='pending' || r.status==='running'); }
  insertResult(runId: string, res: any){ this.data.results[runId] = res; this.save(); }
}

export default JsonDB;
