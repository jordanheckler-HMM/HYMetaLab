#!/usr/bin/env ts-node
import { discoverInstances } from '../experiments/runner/discover';
import { planAll } from '../experiments/runner/plan';
import { dispatchAll } from '../experiments/runner/dispatch';
import { collectLoop } from '../experiments/runner/collect';
import { summarizeAll } from '../experiments/runner/summarize';

async function main(){
  console.log('Discovering instances...');
  const inst = await discoverInstances();
  console.log('Planning runs...');
  await planAll();
  console.log('Dispatching...');
  await dispatchAll();
  console.log('Collecting (single pass)...');
  await collectLoop();
  console.log('Summarizing...');
  await summarizeAll();
  console.log('Done.');
}

if (require.main === module) main().catch(e=>{console.error(e);process.exit(1)});
