/**
 * Voyag V8 CLI for TypeScript bindings.
 *
 * Usage:
 *   node dist/cli.js <model-dir> "打开车窗，同时把空调调到24度"
 */
import { VoyagPredictor } from "./index.js";

async function main(): Promise<void> {
  const modelDir = process.argv[2];
  const utterance = process.argv[3];
  if (!modelDir || !utterance) {
    console.error("usage: node dist/cli.js <model-dir> <utterance>");
    process.exit(2);
  }

  const predictor = await VoyagPredictor.create(modelDir);
  console.log(`device: ${predictor.device}`);
  const result = await predictor.predict(utterance);
  console.log(JSON.stringify(result, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
