import 'package:flutter/material.dart';

import 'api_client.dart';

/// This page is shared by the GitHub Pages iPhone PWA and the Android build.
/// The authenticated API runs the same exposure functions as Streamlit.
class DiffusionOlsPage extends StatefulWidget {
  const DiffusionOlsPage({
    required this.apiUrl,
    required this.identityToken,
    super.key,
  });

  final String Function() apiUrl;
  final String Function() identityToken;

  @override
  State<DiffusionOlsPage> createState() => _DiffusionOlsPageState();
}

class _DiffusionOlsPageState extends State<DiffusionOlsPage> {
  final _tickers = TextEditingController(text: 'AAPL, MSFT, NVDA, GOOGL, AMZN');
  final _historyStart = TextEditingController(text: '2000-01-01');
  final _riskFree = TextEditingController(text: '4.0');
  final _window = TextEditingController(text: '120');
  final _gamma = TextEditingController(text: '5.0');
  final _cap = TextEditingController(text: '0.4');
  final _cost = TextEditingController(text: '10.0');
  final _responseScale = TextEditingController(text: '1.0');
  final _fixedB = TextEditingController(text: '1.0');
  final _grid = TextEditingController(text: '50, 60, 70, 89, 90, 91, 92, 93, 94, 95');
  final _validation = TextEditingController(text: '24');
  final _oosStart = TextEditingController(text: '2019-01-01');
  String _period = '1 week';
  String _mode = 'Fixed b';
  bool _loading = false;
  String? _error;
  Map<String, dynamic>? _result;

  @override
  void dispose() {
    for (final controller in [
      _tickers, _historyStart, _riskFree, _window, _gamma, _cap, _cost,
      _responseScale, _fixedB, _grid, _validation, _oosStart,
    ]) {
      controller.dispose();
    }
    super.dispose();
  }

  Widget _field(
    String label,
    TextEditingController controller, {
    bool date = false,
    bool text = false,
  }) => Padding(
    padding: const EdgeInsets.only(top: 12),
    child: TextField(
      controller: controller,
      keyboardType: date
          ? TextInputType.datetime
          : text
              ? TextInputType.text
              : const TextInputType.numberWithOptions(decimal: true),
      decoration: InputDecoration(labelText: label, border: const OutlineInputBorder()),
    ),
  );

  Future<void> _run() async {
    FocusScope.of(context).unfocus();
    final tickers = _tickers.text
        .split(RegExp(r'[,\s]+'))
        .map((ticker) => ticker.trim().toUpperCase())
        .where((ticker) => ticker.isNotEmpty)
        .toSet()
        .toList();
    final grid = _grid.text.split(',').map((v) => double.tryParse(v.trim())).toList();
    final window = int.tryParse(_window.text.trim());
    final validation = int.tryParse(_validation.text.trim());
    final riskFree = double.tryParse(_riskFree.text.trim());
    final gamma = double.tryParse(_gamma.text.trim());
    final cap = double.tryParse(_cap.text.trim());
    final cost = double.tryParse(_cost.text.trim());
    final scale = double.tryParse(_responseScale.text.trim());
    final b = double.tryParse(_fixedB.text.trim());
    final historyStart = DateTime.tryParse(_historyStart.text.trim());
    final oosStart = DateTime.tryParse(_oosStart.text.trim());
    if (tickers.isEmpty || tickers.length > 10 || historyStart == null ||
        oosStart == null || window == null || riskFree == null ||
        gamma == null || cap == null || cost == null || scale == null ||
        (_mode == 'Fixed b' && b == null) ||
        (_mode == 'Past-only validation' &&
            (validation == null || grid.isEmpty || grid.contains(null)))) {
      setState(() => _error = 'Enter valid dates, tickers, and numeric settings.');
      return;
    }
    setState(() {
      _loading = true;
      _error = null;
      _result = null;
    });
    try {
      final response = await PortfolioApiClient(
        baseUrl: widget.apiUrl(),
        identityToken: widget.identityToken(),
      ).diffusionOlsExposure({
        'tickers': tickers,
        'start_date': _historyStart.text.trim(),
        'holding_period': _period,
        'annual_risk_free_percent': riskFree,
        'window': window,
        'gamma': gamma,
        'cap': cap,
        'cost_bps': cost,
        'response_scale': scale,
        'tuning_mode': _mode,
        'fixed_b': b ?? 1.0,
        'b_grid': grid.whereType<double>().toList(),
        'validation': validation ?? 24,
        'oos_start': _oosStart.text.trim(),
      });
      if (mounted) setState(() => _result = response);
    } on ApiException catch (error) {
      if (mounted) setState(() => _error = error.message);
    } catch (error) {
      if (mounted) setState(() => _error = 'Unexpected error: $error');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) => ListView(
    padding: const EdgeInsets.fromLTRB(12, 4, 12, 24),
    children: [
      Text('Diffusion OLS: Market Exposure',
          style: Theme.of(context).textTheme.headlineSmall),
      const SizedBox(height: 6),
      const Text('Forecast excess returns for your stocks and allocate between '
          'risky assets and cash. No trades are placed.'),
      _field('Ticker symbols', _tickers, text: true),
      _field('History starts (YYYY-MM-DD)', _historyStart, date: true),
      const SizedBox(height: 12),
      DropdownButtonFormField<String>(
        initialValue: _period,
        isExpanded: true,
        decoration: const InputDecoration(
          labelText: 'Holding / rebalance period', border: OutlineInputBorder()),
        items: ['1 week', '2 weeks', '1 month', '2 months', '3 months']
            .map((value) => DropdownMenuItem(value: value, child: Text(value)))
            .toList(),
        onChanged: (value) { if (value != null) setState(() => _period = value); },
      ),
      _field('Annual risk-free rate (%)', _riskFree),
      const SizedBox(height: 20),
      Text('Forecast and exposure settings',
          style: Theme.of(context).textTheme.titleLarge),
      _field('Rolling estimation window', _window),
      _field('Risk aversion γ', _gamma),
      _field('Maximum weight per risky asset', _cap),
      _field('Trading cost (bps per turnover)', _cost),
      _field('Diffusion response scale', _responseScale),
      const SizedBox(height: 12),
      DropdownButtonFormField<String>(
        initialValue: _mode,
        decoration: const InputDecoration(
          labelText: 'Diffusion b selection', border: OutlineInputBorder()),
        items: ['Fixed b', 'Past-only validation']
            .map((value) => DropdownMenuItem(value: value, child: Text(value)))
            .toList(),
        onChanged: (value) { if (value != null) setState(() => _mode = value); },
      ),
      if (_mode == 'Fixed b')
        _field('Fixed b (a = b / window)', _fixedB)
      else ...[
        _field('Candidate b values, comma separated', _grid, text: true),
        _field('Inner validation periods', _validation),
      ],
      _field('Backtest start date (YYYY-MM-DD)', _oosStart, date: true),
      const SizedBox(height: 16),
      FilledButton.icon(
        onPressed: _loading ? null : _run,
        icon: _loading
            ? const SizedBox.square(
                dimension: 18, child: CircularProgressIndicator(strokeWidth: 2))
            : const Icon(Icons.play_arrow),
        label: Text(_loading ? 'Running exposure analysis…' : 'Run Diffusion OLS exposure analysis'),
      ),
      if (_error != null) Card(
        color: Theme.of(context).colorScheme.errorContainer,
        child: Padding(padding: const EdgeInsets.all(14), child: Text(_error!)),
      ),
      if (_result != null) _ExposureResult(result: _result!),
    ],
  );
}

Map<String, dynamic> _asMap(dynamic value) =>
    value is Map<String, dynamic> ? value : <String, dynamic>{};

List<Map<String, dynamic>> _rows(dynamic value) => value is List
    ? value.whereType<Map<String, dynamic>>().toList()
    : <Map<String, dynamic>>[];

String _pct(dynamic value, {int digits = 2}) => value is num
    ? '${(value * 100).toStringAsFixed(digits)}%'
    : '—';

String _dec(dynamic value, {int digits = 3}) => value is num
    ? value.toStringAsFixed(digits)
    : '—';

class _ExposureResult extends StatelessWidget {
  const _ExposureResult({required this.result});

  final Map<String, dynamic> result;

  @override
  Widget build(BuildContext context) {
    final data = _asMap(result['data']);
    final recommendation = _asMap(result['recommendation']);
    final assets = _rows(recommendation['assets']);
    final summary = _rows(result['summary']);
    final rolling = _rows(result['rolling_results']);
    final forecasts = _rows(result['rolling_forecasts']);
    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      const SizedBox(height: 16),
      Text('Next-period exposure', style: Theme.of(context).textTheme.titleLarge),
      Text('Data through ${data['data_through'] ?? '—'} • '
          '${data['observations'] ?? '—'} observations • '
          '${data['predictors'] ?? '—'} predictors'),
      _ExposureLine('Selected b', _dec(recommendation['selected_b'])),
      _ExposureLine('a = b / window', _dec(recommendation['a'], digits: 6)),
      _ExposureLine('Cash weight', _pct(recommendation['cash_weight'])),
      for (final asset in assets) Card(
        child: Padding(padding: const EdgeInsets.all(12), child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('${asset['asset']}', style: Theme.of(context).textTheme.titleMedium),
            _ExposureLine('Forecast excess return', _pct(asset['forecast_excess_return'], digits: 4)),
            _ExposureLine('Estimated variance', _dec(asset['estimated_variance'], digits: 8)),
            _ExposureLine('Risky weight', _pct(asset['risky_weight'])),
          ],
        )),
      ),
      const SizedBox(height: 16),
      Text('Chronological out-of-sample test',
          style: Theme.of(context).textTheme.titleLarge),
      Text('${data['evaluation_start'] ?? '—'} to ${data['evaluation_end'] ?? '—'} • '
          '${data['periods_per_year'] ?? '—'} periods per year'),
      const Text('All performance figures include the selected trading cost.'),
      for (final row in summary) Card(
        child: Padding(padding: const EdgeInsets.all(12), child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('${row['Method']}', style: Theme.of(context).textTheme.titleMedium),
            _ExposureLine('Periods', '${row['Periods']}'),
            _ExposureLine('Total return', _pct(row['Total return'])),
            _ExposureLine('CAGR', _pct(row['CAGR'])),
            _ExposureLine('Annualized volatility', _pct(row['Annualized volatility'])),
            _ExposureLine('Excess Sharpe', _dec(row['Excess Sharpe'])),
            _ExposureLine('Annualized MV excess', _pct(row['Annualized MV excess'])),
            _ExposureLine('Maximum drawdown', _pct(row['Maximum drawdown'])),
            _ExposureLine('Average turnover', _pct(row['Average turnover'])),
            _ExposureLine('Forecast MSE', _dec(row['Forecast MSE'], digits: 8)),
          ],
        )),
      ),
      ExpansionTile(
        title: const Text('Rolling OOS detail'),
        children: [
          for (final row in rolling.where((r) =>
              r['method'] == 'Diffusion OLS' || r['method'] == 'OLS'))
            Card(child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text("${row['method']} • ${row['date']?.toString().split('T').first ?? '—'}",
                    style: Theme.of(context).textTheme.titleSmall),
                _ExposureLine('Gross return', _pct(row['gross_return'], digits: 3)),
                _ExposureLine('Turnover', _pct(row['turnover'])),
                _ExposureLine('Trading cost', _pct(row['cost_fraction'], digits: 3)),
                _ExposureLine('Net return', _pct(row['net_return'], digits: 3)),
                _ExposureLine('Cash weight', _pct(row['cash_weight'])),
                for (final asset in assets)
                  _ExposureLine('${asset['asset']} weight',
                      _pct(row['weight_${asset['asset']}'])),
              ]),
            )),
        ],
      ),
      ExpansionTile(
        title: const Text('Forecast detail and selected b'),
        children: [
          for (final row in forecasts)
            Card(child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text("${row['asset']} • ${row['date']?.toString().split('T').first ?? '—'}",
                    style: Theme.of(context).textTheme.titleSmall),
                _ExposureLine('Actual excess return', _pct(row['actual_excess'], digits: 4)),
                _ExposureLine('Diffusion OLS forecast', _pct(row['diffusion_ols'], digits: 4)),
                _ExposureLine('OLS forecast', _pct(row['ols'], digits: 4)),
                _ExposureLine('Historical mean forecast', _pct(row['historical_mean'], digits: 4)),
                _ExposureLine('Selected b', _dec(row['selected_b'])),
                _ExposureLine('a = b / window', _dec(row['a'], digits: 6)),
              ]),
            )),
        ],
      ),
      const Padding(
        padding: EdgeInsets.all(8),
        child: Text('Historical research simulation; past performance does not guarantee future results.',
            textAlign: TextAlign.center, style: TextStyle(fontSize: 12)),
      ),
    ]);
  }
}

class _ExposureLine extends StatelessWidget {
  const _ExposureLine(this.label, this.value);
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 3),
    child: Row(children: [
      Expanded(child: Text(label)),
      const SizedBox(width: 12),
      Text(value, style: const TextStyle(fontWeight: FontWeight.w600)),
    ]),
  );
}
