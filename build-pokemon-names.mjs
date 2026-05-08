// Fetch all 1025 Pokemon species from PokeAPI and build a structured
// English -> { ja, ja-Hrkt, zh-Hant, zh-Hans, ko, ... } map.
// Run: node build-pokemon-names.mjs
import fs from 'node:fs/promises';

const TOTAL = 1025;
const BATCH = 25;
const OUT = '/Users/wuhonchi/Documents/poke-lookup/pokemon_names.json';
// Lowercase PokeAPI codes (verified earlier)
const LANGS = ['ja-hrkt', 'ja', 'zh-hant', 'zh-hans', 'ko', 'fr', 'de', 'es', 'it'];

async function fetchSpecies(id) {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const r = await fetch(`https://pokeapi.co/api/v2/pokemon-species/${id}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } catch {
      if (attempt === 2) return null;
      await new Promise(res => setTimeout(res, 500 * (attempt + 1)));
    }
  }
}

const result = {};
let processed = 0;

for (let start = 1; start <= TOTAL; start += BATCH) {
  const ids = [];
  for (let i = start; i < start + BATCH && i <= TOTAL; i++) ids.push(i);
  const data = await Promise.all(ids.map(fetchSpecies));

  for (const sp of data) {
    if (!sp) continue;
    const en = sp.names.find(n => n.language.name === 'en')?.name;
    if (!en) continue;
    const entry = {};
    for (const lc of LANGS) {
      const e = sp.names.find(n => n.language.name === lc);
      if (e?.name) entry[lc] = e.name;
    }
    if (Object.keys(entry).length > 0) result[en] = entry;
  }
  processed += ids.length;
  process.stdout.write(`\r  fetched ${processed}/${TOTAL}`);
}
process.stdout.write('\n');

await fs.writeFile(OUT, JSON.stringify(result, null, 2));
const stat = await fs.stat(OUT);
console.log(`Wrote ${OUT}: ${Object.keys(result).length} pokemon, ${(stat.size / 1024).toFixed(1)} KB`);

console.log('Sanity probes:');
['Charizard', 'Pikachu', 'Mewtwo', 'Eevee'].forEach(en => {
  const e = result[en];
  console.log(`  ${en}:`, e ? JSON.stringify(e) : '(missing)');
});
