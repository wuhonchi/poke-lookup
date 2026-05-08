// Fetch all TCGdex EN sets' release dates → en_set_dates.json
// Run: node build-en-set-dates.mjs
import fs from 'node:fs/promises';

const list = await (await fetch('https://api.tcgdex.net/v2/en/sets')).json();
console.log(`Fetching ${list.length} set details for release dates...`);

const dates = {};
let done = 0;
const BATCH = 20;
for (let i = 0; i < list.length; i += BATCH) {
  const slice = list.slice(i, i + BATCH);
  const results = await Promise.all(slice.map(async s => {
    try {
      const r = await fetch(`https://api.tcgdex.net/v2/en/sets/${s.id}`);
      if (!r.ok) return [s.id, null];
      const d = await r.json();
      return [s.id, d.releaseDate || null];
    } catch { return [s.id, null]; }
  }));
  for (const [id, date] of results) if (date) dates[id] = date;
  done += slice.length;
  process.stdout.write(`\r  fetched ${done}/${list.length}`);
}
process.stdout.write('\n');
await fs.writeFile('en_set_dates.json', JSON.stringify(dates));
console.log(`Wrote en_set_dates.json: ${Object.keys(dates).length} sets`);
