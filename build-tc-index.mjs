// Process /tmp/ptcg-db/data_tc/ into a slim cards_tc.json index for the app.
// Output: array of { name, img, set, number, url }, ~2 MB JSON.
// Run: node build-tc-index.mjs
import fs from 'node:fs/promises';
import path from 'node:path';

const SRC = '/tmp/ptcg-db/data_tc';
const OUT = '/Users/wuhonchi/Documents/poke-lookup/cards_tc.json';

async function* walkJson(dir) {
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* walkJson(p);
    else if (entry.isFile() && entry.name.endsWith('.json')) yield p;
  }
}

const cards = [];
let count = 0;
for await (const file of walkJson(SRC)) {
  try {
    const c = JSON.parse(await fs.readFile(file, 'utf8'));
    if (!c.img || !c.name) continue;
    cards.push({
      name: c.name,
      img: c.img,
      set: c.set_name || '',
      number: c.number || '',
      url: c.url || '',
    });
    count++;
    if (count % 5000 === 0) process.stdout.write(`\r  processed ${count}`);
  } catch {
    // skip broken file
  }
}
process.stdout.write(`\r  processed ${count}\n`);

await fs.writeFile(OUT, JSON.stringify(cards));
const stat = await fs.stat(OUT);
console.log(`Wrote ${OUT}: ${cards.length} cards, ${(stat.size / 1024 / 1024).toFixed(2)} MB`);

const probes = ['噴火龍', '皮卡丘', '伊布', '超夢'];
for (const p of probes) {
  const hits = cards.filter(c => c.name.includes(p));
  console.log(`  '${p}' → ${hits.length} cards`);
}
