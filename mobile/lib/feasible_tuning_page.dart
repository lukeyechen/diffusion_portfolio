import 'package:flutter/material.dart';
import 'api_client.dart';

/// Shared by the iPhone PWA and Android.
class FeasibleTuningPage extends StatefulWidget {
  const FeasibleTuningPage({required this.apiUrl, required this.identityToken, super.key});
  final String Function() apiUrl;
  final String Function() identityToken;

  @override
  State<FeasibleTuningPage> createState() => _FeasibleTuningPageState();
}

class _FeasibleTuningPageState extends State<FeasibleTuningPage> {
  final _tickers = TextEditingController(text: 'AAPL, MSFT, NVDA, GOOGL, AMZN');
  final _history = TextEditingController(text: '2000-01-01');
  final _window = TextEditingController(text: '520');
  final _gamma = TextEditingController(text: '3');
  final _cap = TextEditingController(text: '0.4');
  final _cost = TextEditingController(text: '25');
  final _penalty = TextEditingController(text: '25');
  final _rebalance = TextEditingController(text: '50');
  final _m = TextEditingController(text: '500');
  final _beta = TextEditingController(text: '1');
  final _steps = TextEditingController(text: '100');
  final _rf = TextEditingController(text: '0');
  final _cs = TextEditingController(text: '0.25,0.5,1,2,4');
  final _eps = TextEditingController(text: '0.001,0.01,0.05,0.10,0.25');
  late final TextEditingController _start;
  late final TextEditingController _end;
  String _period = '1 week';
  bool _loading = false;
  String? _error;
  Map<String, dynamic>? _result;

  String _date(DateTime d) => '${d.year.toString().padLeft(4, '0')}-${d.month.toString().padLeft(2, '0')}-${d.day.toString().padLeft(2, '0')}';

  @override
  void initState() {
    super.initState();
    final today = DateTime.now();
    final targetMonth = DateTime(today.year, today.month - 6, 1);
    final lastDay = DateTime(targetMonth.year, targetMonth.month + 1, 0).day;
    _start = TextEditingController(text: _date(DateTime(targetMonth.year, targetMonth.month, today.day > lastDay ? lastDay : today.day)));
    _end = TextEditingController(text: _date(today));
  }

  @override
  void dispose() {
    for (final c in [_tickers, _history, _window, _gamma, _cap, _cost, _penalty, _rebalance, _m, _beta, _steps, _rf, _cs, _eps, _start, _end]) {
      c.dispose();
    }
    super.dispose();
  }

  Widget _field(String label, TextEditingController c) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 6),
    child: TextField(controller: c, decoration: InputDecoration(labelText: label, border: const OutlineInputBorder())),
  );

  Future<void> _run() async {
    FocusScope.of(context).unfocus();
    setState(() { _loading = true; _error = null; _result = null; });
    try {
      final response = await PortfolioApiClient(baseUrl: widget.apiUrl(), identityToken: widget.identityToken()).feasibleTuning({
        'tickers': _tickers.text.split(RegExp(r'[,\s]+')).where((v) => v.isNotEmpty).toList(),
        'start_date': _history.text.trim(), 'oos_start': _start.text.trim(), 'end_date': _end.text.trim(),
        'holding_period': _period, 'lookback': int.parse(_window.text),
        'gamma': double.parse(_gamma.text), 'max_long_weight': double.parse(_cap.text),
        'transaction_cost_bps': double.parse(_cost.text), 'turnover_penalty_bps': double.parse(_penalty.text),
        'synthetic_equivalent_m': int.parse(_m.text), 'beta': double.parse(_beta.text), 'reverse_steps': int.parse(_steps.text),
        'rebalance_percent': double.parse(_rebalance.text), 'annual_risk_free_percent': double.parse(_rf.text),
        'cs': _cs.text.split(',').map((v) => double.parse(v.trim())).toList(),
        'epsilons': _eps.text.split(',').map((v) => double.parse(v.trim())).toList(),
      });
      if (mounted) setState(() => _result = response);
    } on FormatException {
      if (mounted) setState(() => _error = 'Enter valid numbers and comma-separated candidate grids.');
    } on Exception catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Widget _table(String title, dynamic value, {bool percent = false}) {
    final rows = (value as List).cast<Map<String, dynamic>>();
    if (rows.isEmpty) return const SizedBox.shrink();
    final columns = rows.first.keys.toList();
    String display(String key, dynamic v) {
      if (v == null) return '—';
      if (v is num) {
        if (percent && !['Periods', 'Sharpe', 'n'].contains(key)) return '${(100 * v).toStringAsFixed(2)}%';
        return v.toStringAsPrecision(6);
      }
      return v.toString();
    }
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Padding(padding: const EdgeInsets.only(top: 20), child: Text(title, style: Theme.of(context).textTheme.titleMedium)),
      SingleChildScrollView(scrollDirection: Axis.horizontal, child: DataTable(
        columns: columns.map((key) => DataColumn(label: Text(key))).toList(),
        rows: rows.map((row) => DataRow(cells: columns.map((key) => DataCell(Text(display(key, row[key])))).toList())).toList(),
      )),
    ]);
  }

  @override
  Widget build(BuildContext context) => ListView(padding: const EdgeInsets.all(16), children: [
    Text('Feasible Tuning', style: Theme.of(context).textTheme.headlineSmall),
    const Text('Run automatically selects c and ε using data before the backtest starts. Classical MV, Trace tuning and Old Portfolio use the same estimation window.'),
    _field('Tickers', _tickers),
    _field('Data history starts (YYYY-MM-DD)', _history),
    DropdownButtonFormField<String>(
      initialValue: _period, decoration: const InputDecoration(labelText: 'Holding period'),
      items: ['1 week', '2 weeks', '1 month', '2 months', '3 months'].map((v) => DropdownMenuItem(value: v, child: Text(v))).toList(),
      onChanged: _loading ? null : (v) => setState(() {
        _period = v!;
        _window.text = {'1 week': '520', '2 weeks': '260', '1 month': '120', '2 months': '60', '3 months': '40'}[v]!;
        _rebalance.text = v == '1 week' ? '50' : '100';
      }),
    ),
    _field('Main estimation window', _window),
    _field('Risk aversion γ', _gamma),
    _field('Maximum asset weight', _cap),
    _field('Backtest start date (YYYY-MM-DD)', _start),
    _field('End date (YYYY-MM-DD)', _end),
    ExpansionTile(title: const Text('Calibration and trading settings'), children: [
      _field('c candidates', _cs), _field('ε candidates', _eps),
      _field('Annual risk-free return (%)', _rf), _field('Trading cost (bps)', _cost),
      _field('Turnover penalty (bps)', _penalty), _field('Rebalance step (%)', _rebalance),
      _field('Old-method synthetic-equivalent M', _m), _field('Constant β', _beta), _field('Old-method reverse SDE steps', _steps),
    ]),
    const SizedBox(height: 12),
    FilledButton(onPressed: _loading ? null : _run, child: Text(_loading ? 'Calibrating and replaying…' : 'Run feasible-tuning backtest')),
    if (_loading) const LinearProgressIndicator(),
    if (_error != null) Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
    if (_result != null) ...[
      Text('Selected c = ${_result!['selected_c']}; ε = ${_result!['selected_epsilon']}'),
      Text('Calibration through ${_result!['calibration_through']}; data through ${_result!['data_through']}'),
      _table('Historical results', _result!['summary'], percent: true),
      _table('Raw optimal targets', _result!['raw_allocations'], percent: true),
      _table('Final allocations', _result!['allocations'], percent: true),
      _table('Latest tuning values', _result!['tuning']),
    ],
  ]);
}
