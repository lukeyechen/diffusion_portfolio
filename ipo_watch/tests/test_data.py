import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ipo_data import parse_directory, merge_snapshot, refresh, read_snapshot, validate_row

class DataTests(unittest.TestCase):
    def test_realistic_html_and_missing_price(self):
        html = '<table><tr><th>Date</th></tr><tr><td>Jan 9, 2026</td><td><a>ABC</a></td><td>A &amp; B Space Systems</td><td>$1,234.50</td></tr><tr><td>Jan 8, 2026</td><td>DEF</td><td>Other Acquisition Corp.</td><td>-</td></tr></table>'
        rows = parse_directory(html, 2026)
        self.assertEqual(rows[0]['name'], 'A & B Space Systems')
        self.assertEqual(rows[0]['kind'], 'Listing')
        self.assertEqual(rows[0]['offer'], 1234.5)
        self.assertIsNone(rows[1]['offer'])
        self.assertEqual(rows[1]['kind'], 'SPAC')
    def test_merge_preserves_history_and_deduplicates(self):
        old={'date':'2025-01-01','ticker':'OLD','name':'Old Co','offer':10}
        new={'date':'2026-01-01','ticker':'NEW','name':'New Co','offer':20}
        result=merge_snapshot({'rows':[old,new]},[new,new],'https://example.com')
        self.assertEqual(len(result['rows']),2)
        self.assertEqual(result['rows'][0]['ticker'],'NEW')
    def test_outage_does_not_overwrite_snapshot(self):
        snapshot={'rows':[{'date':'2026-01-01','ticker':'ABC','name':'A','offer':10}], 'updated_at':'2026-01-01T00:00:00+00:00','source':'https://example.com'}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ipos.json';path.write_text(json.dumps(snapshot));before=path.read_bytes()
            with patch('ipo_data.fetch_rows',side_effect=OSError('outage')):
                with self.assertRaises(OSError):refresh(path)
            self.assertEqual(path.read_bytes(),before)
    def test_invalid_data_rejected(self):
        with self.assertRaises(ValueError):parse_directory('<html>Unavailable</html>',2026)
        with self.assertRaises(ValueError):validate_row({'date':'2099-01-01','ticker':'BAD','name':'Future','offer':10})
    def test_bundled_snapshot(self):
        data=read_snapshot();keys=[(r['date'],r['ticker']) for r in data['rows']]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertTrue(any(r['ticker']=='NARA' for r in data['rows']))

if __name__=='__main__':unittest.main()
