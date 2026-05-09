// Fetch Yahoo Auctions JP sold prices for JP cards (raw + PSA10).
// For each card, extract:
//   - last 10 transactions (title, price, endTime, auctionId)
//   - last 24h transactions (subset of last 10)
//   - 120-day avg + sample size from meta description
// 2 second delay between requests. Save progress incrementally so killable/resumable.
//
// Usage:
//   node scripts/fetch-jp-prices.mjs           # default 1000 cards
//   node scripts/fetch-jp-prices.mjs 100       # only 100 cards
//   node scripts/fetch-jp-prices.mjs 1000 50   # 1000 cards starting at index 50

import fs from 'node:fs/promises';
import path from 'node:path';

const ROOT = '/Users/wuhonchi/Documents/poke-lookup';
const INPUT = path.join(ROOT, 'cards_jp.json');
const OUTPUT = path.join(ROOT, 'prices_jp.json');
const STATE = path.join(ROOT, 'prices_jp.state.json');

const LIMIT = parseInt(process.argv[2] || '1000', 10);
const START = parseInt(process.argv[3] || '0', 10);
const DELAY_MS = 2000;

const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36';
const headers = {
  'User-Agent': UA,
  'Accept-Language': 'ja-JP,ja;q=0.9',
  'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function fetchYahoo(query) {
  // Sort by end date desc to get newest sold listings
  const url = `https://auctions.yahoo.co.jp/closedsearch/closedsearch?p=${encodeURIComponent(query)}&fixed=1&s1=end&o1=d`;
  const r = await fetch(url, { headers });
  if (!r.ok) {
    return { error: `HTTP ${r.status}`, status: r.status };
  }
  const html = await r.text();

  // Extract structured listings from __NEXT_DATA__
  const ndMatch = html.match(/<script[^>]*id="__NEXT_DATA__"[^>]*>(\{.+?\})<\/script>/s);
  let listings = [];
  if (ndMatch) {
    try {
      const data = JSON.parse(ndMatch[1]);
      const items = data?.props?.pageProps?.initialState?.search?.items?.listing?.items || [];
      listings = items.map((it) => ({
        auctionId: it.auctionId,
        title: it.title,
        price: it.price,
        endTime: it.endTime,
        bidCount: it.bidCount,
        buyNowPrice: it.buyNowPrice ?? null,
      }));
    } catch (e) {
      // ignore parse error, fall through
    }
  }

  // Extract aggregate stats from meta description
  const metaMatch = html.match(/(?:約)?\s*([\d,]+)\s*件の落札価格は平均\s*([\d,]+)/);
  const aggregate = metaMatch
    ? { sampleSize120d: +metaMatch[1].replace(/,/g, ''), avg120d: +metaMatch[2].replace(/,/g, '') }
    : null;

  return { listings, aggregate, fetchedAt: new Date().toISOString() };
}

// Filter listings whose title actually contains the card name (and PSA10 keyword if PSA10 query).
// Yahoo's full-text search OR-falls-back when keywords don't co-occur; meta-description avg
// becomes garbage (e.g. 119,000+ irrelevant items). Client-side filtering keeps only true matches.
function filterRelevant(listings, cardName, requirePsa10) {
  return listings.filter((it) => {
    const title = it.title || '';
    if (!title.includes(cardName)) return false;
    if (requirePsa10 && !/PSA\s*10|PSA10/i.test(title)) return false;
    return true;
  });
}

function summarize(result, cardName, requirePsa10) {
  if (result?.error) return result;
  const { listings = [], aggregate, fetchedAt } = result;
  const filtered = filterRelevant(listings, cardName, requirePsa10);
  const sortedByEnd = filtered.slice().sort(
    (a, b) => new Date(b.endTime).getTime() - new Date(a.endTime).getTime()
  );
  const last10 = sortedByEnd.slice(0, 10);
  const cutoff = Date.now() - 24 * 3600 * 1000;
  const last24h = sortedByEnd.filter((it) => new Date(it.endTime).getTime() >= cutoff);

  // Compute our own avg from filtered listings (more trustworthy than Yahoo meta)
  const prices = last10.map((it) => it.price).filter((p) => p > 0);
  const avgRecent = prices.length ? Math.round(prices.reduce((a, b) => a + b, 0) / prices.length) : null;

  // Trust Yahoo meta only if filtered count seems consistent (meta sample <= 5x filtered count)
  // Otherwise meta is likely OR-fallback garbage.
  const metaTrustworthy = aggregate && filtered.length > 0 && aggregate.sampleSize120d <= filtered.length * 50;

  return {
    fetchedAt,
    rawListings: listings.length,        // raw Yahoo result count (may include garbage)
    relevantListings: filtered.length,   // after client-side filter
    avgRecent,                           // our avg from last 10 filtered
    last10,
    last24h,
    avg120d: metaTrustworthy ? aggregate.avg120d : null,
    sampleSize120d: metaTrustworthy ? aggregate.sampleSize120d : null,
    metaUntrusted: !metaTrustworthy && aggregate ? aggregate : null,
  };
}

function buildQuery(card, withPsa10 = false) {
  // Query strategy: include card name + local number to disambiguate variants
  // e.g., "リザードンex 006" matches listings with both terms
  const local = (card.number || '').split('/')[0].replace(/^0+/, '') || card.number;
  const parts = [card.name];
  if (local) parts.push(local);
  if (withPsa10) parts.push('PSA10');
  return parts.join(' ');
}

async function main() {
  const allCards = JSON.parse(await fs.readFile(INPUT, 'utf8'));
  console.log(`Loaded ${allCards.length} JP cards from ${INPUT}`);

  // Sort by jp_id desc (newest first) so we cover modern cards first
  allCards.sort((a, b) => (b.jp_id || 0) - (a.jp_id || 0));

  const slice = allCards.slice(START, START + LIMIT);
  console.log(`Processing ${slice.length} cards (index ${START} to ${START + slice.length - 1}), 2s delay between requests`);
  console.log(`Estimated time: ${((slice.length * 2 * DELAY_MS) / 1000 / 60).toFixed(1)} min\n`);

  // Resume support: load existing prices file
  let prices = {};
  try {
    prices = JSON.parse(await fs.readFile(OUTPUT, 'utf8'));
    console.log(`Resuming with ${Object.keys(prices).length} existing entries`);
  } catch {
    console.log('No existing prices file, starting fresh');
  }

  const startTime = Date.now();
  let okCount = 0, errCount = 0, blockCount = 0;
  let lastSave = Date.now();

  for (let i = 0; i < slice.length; i++) {
    const card = slice[i];
    const cardKey = `${card.set}-${card.number}`;

    // Skip if already fetched recently
    if (prices[cardKey]?.fetchedAt) {
      const age = Date.now() - new Date(prices[cardKey].fetchedAt).getTime();
      if (age < 23 * 3600 * 1000) {
        if (i % 50 === 0) process.stdout.write(`\r[${i + 1}/${slice.length}] cached: ${cardKey}      `);
        continue;
      }
    }

    // Fetch raw
    const rawQ = buildQuery(card, false);
    const rawResult = await fetchYahoo(rawQ);
    await sleep(DELAY_MS);

    if (rawResult.status === 403 || rawResult.status === 429) {
      blockCount++;
      console.log(`\n⚠️ Yahoo blocked at card ${i + 1}/${slice.length} (${cardKey}): HTTP ${rawResult.status}`);
      console.log(`   Backing off 60s...`);
      await sleep(60000);
      i--; // retry this card
      continue;
    }

    // Fetch PSA10
    const psaQ = buildQuery(card, true);
    const psaResult = await fetchYahoo(psaQ);
    await sleep(DELAY_MS);

    if (psaResult.status === 403 || psaResult.status === 429) {
      blockCount++;
      console.log(`\n⚠️ Yahoo blocked at card ${i + 1} (PSA10 query)`);
      await sleep(60000);
    }

    if (rawResult.error && psaResult.error) {
      errCount++;
    } else {
      okCount++;
    }

    prices[cardKey] = {
      card: { name: card.name, set: card.set, number: card.number, jp_id: card.jp_id },
      raw: summarize(rawResult, card.name, false),
      psa10: summarize(psaResult, card.name, true),
      query: { raw: rawQ, psa10: psaQ },
    };

    // Progress log every 10 cards
    if ((i + 1) % 10 === 0) {
      const elapsed = (Date.now() - startTime) / 1000;
      const rate = (i + 1) / elapsed;
      const eta = ((slice.length - i - 1) / rate / 60).toFixed(1);
      process.stdout.write(
        `\r[${i + 1}/${slice.length}] ok=${okCount} err=${errCount} block=${blockCount} | rate=${rate.toFixed(2)}/s | ETA ${eta}m   `
      );
    }

    // Save every 30 seconds
    if (Date.now() - lastSave > 30000) {
      await fs.writeFile(OUTPUT, JSON.stringify(prices));
      lastSave = Date.now();
    }
  }

  // Final save
  await fs.writeFile(OUTPUT, JSON.stringify(prices, null, 2));
  console.log(`\n\nDone. Wrote ${OUTPUT}`);
  console.log(`Total entries: ${Object.keys(prices).length}, ok=${okCount}, err=${errCount}, block=${blockCount}`);

  // Quick stats
  let withRaw = 0, withPsa = 0, withBoth = 0;
  for (const k in prices) {
    const e = prices[k];
    const hasRaw = !e.raw.error && (e.raw.avg120d || e.raw.last10?.length);
    const hasPsa = !e.psa10.error && (e.psa10.avg120d || e.psa10.last10?.length);
    if (hasRaw) withRaw++;
    if (hasPsa) withPsa++;
    if (hasRaw && hasPsa) withBoth++;
  }
  console.log(`\nCoverage: ${withRaw} have raw data, ${withPsa} have PSA10 data, ${withBoth} have both`);
}

main().catch((e) => {
  console.error('\nFATAL:', e);
  process.exit(1);
});
