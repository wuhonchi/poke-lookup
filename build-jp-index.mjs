// Process /tmp/ptcg-db/data_jp/ into a slim cards_jp.json index for the app.
// Output: array of { name, img, set, number, jp_id }, ~3-5 MB JSON.
// Run: node build-jp-index.mjs
import fs from 'node:fs/promises';
import path from 'node:path';

const SRC = '/tmp/ptcg-db/data_jp';
const OUT = '/Users/wuhonchi/Documents/poke-lookup/cards_jp.json';

async function* walkJson(dir) {
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* walkJson(p);
    else if (entry.isFile() && entry.name.endsWith('.json')) yield p;
  }
}

// Dedup by image URL (defensive — JP normally has only ~6 dupes).
const seen = new Map();
let count = 0, dupes = 0;
for await (const file of walkJson(SRC)) {
  try {
    const c = JSON.parse(await fs.readFile(file, 'utf8'));
    if (!c.img || !c.name) continue;
    if (seen.has(c.img)) { dupes++; count++; continue; }
    seen.set(c.img, {
      name: c.name,
      img: c.img,
      set: c.set_name || '',
      number: c.number || '',
      jp_id: c.jp_id || null,
    });
    count++;
    if (count % 5000 === 0) process.stdout.write(`\r  processed ${count}`);
  } catch (e) {
    // skip broken file
  }
}
const cards = Array.from(seen.values());
process.stdout.write(`\r  processed ${count}, deduped ${dupes}\n`);

await fs.writeFile(OUT, JSON.stringify(cards));
const stat = await fs.stat(OUT);
console.log(`Wrote ${OUT}: ${cards.length} cards, ${(stat.size / 1024 / 1024).toFixed(2)} MB`);

// Sanity check: count Charizard variants
const riza = cards.filter(c => c.name.includes('リザード'));
console.log(`Sanity: cards containing 'リザード': ${riza.length}`);
console.log(`Sample (first 3):`);
riza.slice(0, 3).forEach(c => console.log(`  ${c.set}/${c.number} ${c.name} → ${c.img}`));
