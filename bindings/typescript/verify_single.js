// Run one prediction via the TS binding and print JSON. Used by the
// Python cross-check script (tmp/verify_ts_vs_python.py).
import { resolve, join } from "node:path";
import { VoyagPredictor } from "./dist/index.js";

const utterance = process.argv[2];
if (!utterance) {
  console.error("usage: node verify_single.js <utterance>");
  process.exit(2);
}

// Model dir = two levels up from bindings/typescript.
const modelDir = resolve(join(import.meta.dirname, "..", ".."));

const predictor = await VoyagPredictor.create(modelDir);
const result = await predictor.predict(utterance);
console.log(JSON.stringify(result));
