// Replace the NAME_ALIASES block in index.html with the contents of aliases.json
import fs from 'node:fs/promises';

const aliases = JSON.parse(await fs.readFile('aliases.json', 'utf8'));
const html = await fs.readFile('index.html', 'utf8');

const startMarker = 'const NAME_ALIASES = {';
const start = html.indexOf(startMarker);
if (start === -1) throw new Error('Could not find NAME_ALIASES start');

let depth = 0;
let i = start + startMarker.length - 1;
for (; i < html.length; i++) {
  if (html[i] === '{') depth++;
  else if (html[i] === '}') {
    depth--;
    if (depth === 0) break;
  }
}
if (depth !== 0) throw new Error('Could not find matching closing brace');
const end = i + 1;
while (html[end] === ';') i = end;
const endWithSemi = html.indexOf(';', end) + 1;

const sortedKeys = Object.keys(aliases).sort();
const lines = sortedKeys.map(k => `  ${JSON.stringify(k)}: ${JSON.stringify(aliases[k])},`);
const replacement = `const NAME_ALIASES = {\n${lines.join('\n')}\n};`;

const out = html.slice(0, start) + replacement + html.slice(endWithSemi);
await fs.writeFile('index.html', out);
console.log(`Injected ${sortedKeys.length} aliases. New file size: ${out.length} bytes`);
