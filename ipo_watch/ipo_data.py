"""Fetch and validate completed U.S. IPO listings; preserve last good data."""
from __future__ import annotations
import argparse
import json
import os
import re
import tempfile
import urllib.request
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / 'data' / 'ipos.json'

class DirectoryParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.row, self.cell = [], None, None
    def handle_starttag(self, tag, attrs):
        if tag == 'tr': self.row = []
        elif tag == 'td' and self.row is not None: self.cell = []
    def handle_data(self, text):
        if self.cell is not None: self.cell.append(text)
    def handle_endtag(self, tag):
        if tag == 'td' and self.cell is not None:
            self.row.append(' '.join(''.join(self.cell).split()))
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row, self.cell = None, None

def validate_row(row):
    listed = date.fromisoformat(row['date'])
    if listed > datetime.now(timezone.utc).date():
        raise ValueError('Future listings must not be imported')
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-^]{0,19}', row['ticker']):
        raise ValueError('Invalid ticker')
    name = str(row['name']).strip()
    if not name or len(name) > 300: raise ValueError('Invalid company name')
    price = row.get('offer')
    if price is not None and (not isinstance(price, (int, float)) or not 0 < price < 1_000_000):
        raise ValueError('Invalid offer price')
    kind = 'SPAC' if re.search(r'acquisition|\bspac\b|merger|equity partners', name, re.I) else 'Listing'
    return {'date': listed.isoformat(), 'ticker': row['ticker'], 'name': name, 'offer': price, 'kind': kind}

def parse_directory(html, year):
    parser = DirectoryParser()
    parser.feed(html)
    result = []
    for cells in parser.rows:
        if len(cells) < 4: continue
        try: listed = datetime.strptime(cells[0], '%b %d, %Y').date()
        except ValueError: continue
        if listed.year != year or listed > datetime.now(timezone.utc).date(): continue
        raw = cells[3].replace('$', '').replace(',', '')
        try: price = float(raw)
        except ValueError: price = None
        result.append(validate_row({'date': listed.isoformat(), 'ticker': cells[1], 'name': cells[2], 'offer': price}))
    if not result: raise ValueError('No completed listings found; the source may be unavailable or its format may have changed')
    return result

def fetch_rows(year):
    source = f'https://stockanalysis.com/ipos/{year}/'
    req = urllib.request.Request(source, headers={'User-Agent': 'IPO-Watch/1.0 (personal IPO directory)', 'Accept': 'text/html'})
    with urllib.request.urlopen(req, timeout=20) as response:
        html = response.read(8_000_000).decode('utf-8')
    return parse_directory(html, year), source

def read_snapshot(path=DATA_PATH):
    snapshot = json.loads(Path(path).read_text())
    snapshot['rows'] = [validate_row(r) for r in snapshot['rows']]
    return snapshot

def merge_snapshot(previous, fresh, source):
    merged = {(r['date'], r['ticker']): validate_row(r) for r in previous['rows']}
    for row in fresh:
        row = validate_row(row)
        merged[(row['date'], row['ticker'])] = row
    return {'rows': sorted(merged.values(), key=lambda r: (r['date'], r['ticker']), reverse=True),
            'updated_at': datetime.now(timezone.utc).isoformat(), 'source': source}

def atomic_save(snapshot, path=DATA_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp', delete=False) as f:
            name = f.name
            json.dump(snapshot, f, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(name, path)
    finally:
        if name and Path(name).exists(): Path(name).unlink()

def refresh(path=DATA_PATH, year=None):
    previous = read_snapshot(path)
    fresh, source = fetch_rows(year or datetime.now(timezone.utc).year)
    snapshot = merge_snapshot(previous, fresh, source)
    atomic_save(snapshot, path)
    return snapshot

if __name__ == '__main__':
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--year', type=int)
    options = args.parse_args()
    try:
        result = refresh(year=options.year)
        print(f"Saved {len(result['rows'])} listings; checked {result['source']}")
    except Exception as error:
        raise SystemExit(f'IPO refresh failed; previous snapshot retained. {error}')
