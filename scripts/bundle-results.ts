#!/usr/bin/env ts-node
import { bundleAll } from '../experiments/runner/bundle';

if (require.main===module) bundleAll().then(p=>console.log(p)).catch(e=>{console.error(e);process.exit(1)});
