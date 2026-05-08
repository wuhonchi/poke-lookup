// Scrape pokemon-card.com/ex/{code}/ pages for JP set release dates + full names.
// Output: jp_set_info.json = { "SV2a": { date: "2023-06-16", name: "ポケモンカード151" }, ... }
// Run: node build-jp-set-info.mjs
import fs from 'node:fs/promises';

const cards = JSON.parse(await fs.readFile('cards_jp.json', 'utf8'));
const setCodes = [...new Set(cards.map(c => c.set).filter(Boolean))];
console.log(`Unique JP set codes: ${setCodes.length}`);

const result = {};
const BATCH = 10;
let done = 0;

// pokemon-card.com uses combined codes for paired sets:
//   SV11W + SV11B → sv11
//   M1S + M1L     → m1
//   SV1S + SV1V   → sv1
// So we try the original code first, then progressively strip trailing letters.
function urlVariants(code) {
  const lower = code.toLowerCase();
  const out = [lower];
  // Strip 1 trailing letter (s/v/w/b/d/p/l/a/k/h/m)
  const m1 = lower.match(/^(.*?)([swvbdplakhm])$/);
  if (m1 && /\d/.test(m1[1])) out.push(m1[1]);
  // Strip 2 trailing letters (e.g. sv11bF -> sv11)
  const m2 = lower.match(/^(.*?\d)([a-z]{1,2})$/);
  if (m2) out.push(m2[1]);
  return [...new Set(out)];
}

async function fetchSet(code) {
  const variants = urlVariants(code);
  let html = null;
  let triedUrl = null;
  for (const v of variants) {
    const url = `https://www.pokemon-card.com/ex/${v}/`;
    try {
      const r = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0 poke-lookup-builder' } });
      if (r.ok) { html = await r.text(); triedUrl = url; break; }
    } catch {}
  }
  if (!html) return null;
  try {
    // Extract title — typically "強化拡張パック「ポケモンカード151」｜...."
    const titleMatch = html.match(/<title>([^<]+)<\/title>/);
    let setName = null;
    if (titleMatch) {
      const raw = titleMatch[1].trim();
      const m = raw.match(/「([^」]+)」/);
      setName = m ? m[1] : raw.split('｜')[0].trim();
    }
    // Extract release date — try several strategies, fall back to first date on page
    const allDates = [...html.matchAll(/(20\d{2})年(\d{1,2})月(\d{1,2})日/g)];
    if (!allDates.length) return setName ? { name: setName, date: null } : null;
    // Prefer a date within 200 chars of 発売
    let pick = null;
    for (const m of allDates) {
      const around = html.slice(Math.max(0, m.index - 200), m.index + 200);
      if (around.includes('発売')) { pick = m; break; }
    }
    pick = pick || allDates[0];
    const [, y, mo, d] = pick;
    return { name: setName, date: `${y}-${mo.padStart(2,'0')}-${d.padStart(2,'0')}` };
  } catch { return null; }
}

for (let i = 0; i < setCodes.length; i += BATCH) {
  const slice = setCodes.slice(i, i + BATCH);
  const out = await Promise.all(slice.map(c => fetchSet(c).then(r => [c, r])));
  for (const [code, info] of out) if (info) result[code] = info;
  done += slice.length;
  process.stdout.write(`\r  fetched ${done}/${setCodes.length}, found ${Object.keys(result).length}`);
}
process.stdout.write('\n');

await fs.writeFile('jp_set_info.json', JSON.stringify(result, null, 2));
const stat = await fs.stat('jp_set_info.json');
console.log(`Wrote jp_set_info.json: ${Object.keys(result).length} sets, ${(stat.size/1024).toFixed(1)} KB`);
console.log(`Coverage: ${Object.keys(result).length}/${setCodes.length} = ${(Object.keys(result).length*100/setCodes.length).toFixed(1)}%`);

// Sanity probes
['SV2a', 'M3', 'SV11W', 'SV1', 'BW1-Bb', 'CP6'].forEach(c => {
  console.log(`  ${c}: ${result[c] ? JSON.stringify(result[c]) : '(missing)'}`);
});
