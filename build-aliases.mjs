// Fetch all Pokemon species names from PokeAPI and rebuild NAME_ALIASES in index.html.
// Run: node build-aliases.mjs
import fs from 'node:fs/promises';

const TOTAL = 1025;
const BATCH = 25;
const LANGS = ['ja-hrkt', 'ja', 'zh-hant', 'zh-hans', 'ko', 'fr', 'de', 'es', 'it'];

async function fetchSpecies(id) {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const r = await fetch(`https://pokeapi.co/api/v2/pokemon-species/${id}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } catch (e) {
      if (attempt === 2) {
        console.error(`  failed id=${id}: ${e.message}`);
        return null;
      }
      await new Promise(res => setTimeout(res, 500 * (attempt + 1)));
    }
  }
}

async function main() {
  const aliases = {};
  let processed = 0;

  for (let start = 1; start <= TOTAL; start += BATCH) {
    const ids = [];
    for (let i = start; i < start + BATCH && i <= TOTAL; i++) ids.push(i);
    const results = await Promise.all(ids.map(fetchSpecies));

    for (const sp of results) {
      if (!sp) continue;
      const enEntry = sp.names.find(n => n.language.name === 'en');
      if (!enEntry) continue;
      const englishName = enEntry.name;
      for (const langCode of LANGS) {
        const entry = sp.names.find(n => n.language.name === langCode);
        if (!entry || !entry.name) continue;
        const localized = entry.name.trim();
        if (!localized || localized === englishName) continue;
        if (!aliases[localized]) aliases[localized] = englishName;
      }
    }
    processed += ids.length;
    process.stdout.write(`\r  fetched ${processed}/${TOTAL}`);
  }
  process.stdout.write('\n');

  console.log(`Total alias entries: ${Object.keys(aliases).length}`);

  // Quick sanity check
  const probes = ['噴火龍', 'リザードン', '皮卡丘', 'ピカチュウ', '伊布', 'Glurak'];
  for (const p of probes) {
    console.log(`  ${p} -> ${aliases[p] ?? '(not found)'}`);
  }

  await fs.writeFile('aliases.json', JSON.stringify(aliases, null, 2));
  console.log('Wrote aliases.json');
}

main();
