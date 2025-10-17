#!/usr/bin/env ts-node
import { summarizeAll } from '../experiments/runner/summarize';

if (require.main===module) summarizeAll().catch(e=>{console.error(e);process.exit(1)});
