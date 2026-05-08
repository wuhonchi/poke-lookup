// Fetch TCGdex zh-tw sets list for TC set release dates + full names.
// Output: tc_set_info.json = { "SVI": { date: "...", name: "..." }, ... }
// Run: node build-tc-set-info.mjs
import fs from 'node:fs/promises';

const list = await (await fetch('https://api.tcgdex.net/v2/zh-tw/sets')).json();
console.log(`TCGdex zh-tw sets: ${list.length}`);

const result = {};
let done = 0;
const BATCH = 20;
for (let i = 0; i < list.length; i += BATCH) {
  const slice = list.slice(i, i + BATCH);
  const r = await Promise.all(slice.map(async s => {
    try {
      const r = await fetch(`https://api.tcgdex.net/v2/zh-tw/sets/${s.id}`);
      if (!r.ok) return [s.id, null];
      const d = await r.json();
      return [s.id, { name: d.name || null, date: d.releaseDate || null }];
    } catch { return [s.id, null]; }
  }));
  for (const [id, info] of r) if (info) result[id] = info;
  done += slice.length;
  process.stdout.write(`\r  fetched ${done}/${list.length}`);
}
process.stdout.write('\n');

await fs.writeFile('tc_set_info.json', JSON.stringify(result, null, 2));
console.log(`Wrote tc_set_info.json: ${Object.keys(result).length} sets`);

// Sanity check: how many of our PTCG-DB cards' sets have info?
const cards = JSON.parse(await fs.readFile('cards_tc.json', 'utf8'));
const ourSets = new Set(cards.map(c => c.set).filter(Boolean));
const matched = [...ourSets].filter(s => result[s]);
console.log(`PTCG-DB TC sets covered: ${matched.length}/${ourSets.size} = ${(matched.length*100/ourSets.size).toFixed(1)}%`);

['SVI', 'SV1S', 'SV2a', 'SCC', 'S4'].forEach(c => {
  console.log(`  ${c}: ${result[c] ? JSON.stringify(result[c]) : '(missing)'}`);
});
