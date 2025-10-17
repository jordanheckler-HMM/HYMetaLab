import path from 'path';
import fs from 'fs-extra';
import archiver from 'archiver';

const RESULTS_DIR = path.resolve(process.cwd(),'data','experiments');
const OUT_DIR = path.resolve(process.cwd(),'data','experiments','bundles');

export async function bundleAll(namespace='exp_2025Q4'){
  await fs.ensureDir(OUT_DIR);
  const ts = (new Date()).toISOString().replace(/[:.]/g,'-');
  const zipPath = path.join(OUT_DIR, `${namespace}_${ts}.zip`);
  const output = fs.createWriteStream(zipPath);
  const archive = archiver('zip');
  archive.pipe(output);
  archive.directory(RESULTS_DIR, false);
  await archive.finalize();
  console.log('Created bundle:', zipPath);
  return zipPath;
}

if (require.main===module) bundleAll().catch(e=>{console.error(e);process.exit(1)});
