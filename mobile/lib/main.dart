import 'package:flutter/material.dart';

import 'api_client.dart';

void main() {
  runApp(const DiffusionPortfolioApp());
}

class DiffusionPortfolioApp extends StatelessWidget {
  const DiffusionPortfolioApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Diffusion Portfolio',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xff3157d5)),
        useMaterial3: true,
      ),
      home: const PortfolioHomePage(),
    );
  }
}

class PortfolioHomePage extends StatefulWidget {
  const PortfolioHomePage({super.key});

  @override
  State<PortfolioHomePage> createState() => _PortfolioHomePageState();
}

class _PortfolioHomePageState extends State<PortfolioHomePage> {
  static const _configuredApiUrl = String.fromEnvironment('API_BASE_URL');

  final _apiUrlController = TextEditingController(text: _configuredApiUrl);
  final _tokenController = TextEditingController();
  int _selectedPage = 0;
  bool _showConnection = true;

  @override
  void dispose() {
    _apiUrlController.dispose();
    _tokenController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Diffusion Portfolio'),
        actions: [
          IconButton(
            tooltip: 'Private API connection',
            onPressed: () => setState(() => _showConnection = !_showConnection),
            icon: const Icon(Icons.lock_outline),
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            if (_showConnection)
              _ConnectionCard(
                apiUrlController: _apiUrlController,
                tokenController: _tokenController,
                onDone: () => setState(() => _showConnection = false),
              ),
            Expanded(
              child: IndexedStack(
                index: _selectedPage,
                children: [
                  RecommendationPage(
                    apiUrl: () => _apiUrlController.text,
                    identityToken: () => _tokenController.text,
                  ),
                  BacktestPage(
                    apiUrl: () => _apiUrlController.text,
                    identityToken: () => _tokenController.text,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedPage,
        onDestinationSelected: (value) => setState(() => _selectedPage = value),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.pie_chart_outline),
            selectedIcon: Icon(Icons.pie_chart),
            label: 'Portfolio',
          ),
          NavigationDestination(
            icon: Icon(Icons.show_chart),
            label: 'Backtest',
          ),
        ],
      ),
    );
  }
}

class _ConnectionCard extends StatefulWidget {
  const _ConnectionCard({
    required this.apiUrlController,
    required this.tokenController,
    required this.onDone,
  });

  final TextEditingController apiUrlController;
  final TextEditingController tokenController;
  final VoidCallback onDone;

  @override
  State<_ConnectionCard> createState() => _ConnectionCardState();
}

class _ConnectionCardState extends State<_ConnectionCard> {
  bool _showToken = false;

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.fromLTRB(12, 4, 12, 8),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.verified_user_outlined, size: 20),
                SizedBox(width: 8),
                Text(
                  'Private Google Cloud connection',
                  style: TextStyle(fontWeight: FontWeight.w600),
                ),
              ],
            ),
            const SizedBox(height: 12),
            TextField(
              controller: widget.apiUrlController,
              keyboardType: TextInputType.url,
              autocorrect: false,
              decoration: const InputDecoration(
                labelText: 'API URL',
                hintText: 'https://diffusion-portfolio-api-…run.app',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 10),
            TextField(
              controller: widget.tokenController,
              obscureText: !_showToken,
              autocorrect: false,
              enableSuggestions: false,
              decoration: InputDecoration(
                labelText: 'Temporary Google identity token',
                border: const OutlineInputBorder(),
                suffixIcon: IconButton(
                  tooltip: _showToken ? 'Hide token' : 'Show token',
                  onPressed: () => setState(() => _showToken = !_showToken),
                  icon: Icon(_showToken ? Icons.visibility_off : Icons.visibility),
                ),
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'The token stays in memory only and is erased when the app closes.',
              style: TextStyle(fontSize: 12),
            ),
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerRight,
              child: FilledButton.tonal(
                onPressed: widget.onDone,
                child: const Text('Done'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class RecommendationPage extends StatefulWidget {
  const RecommendationPage({
    required this.apiUrl,
    required this.identityToken,
    super.key,
  });

  final String Function() apiUrl;
  final String Function() identityToken;

  @override
  State<RecommendationPage> createState() => _RecommendationPageState();
}

class _RecommendationPageState extends State<RecommendationPage> {
  final _tickersController =
      TextEditingController(text: 'AAPL, MSFT, NVDA, GOOGL, AMZN');
  final _startController = TextEditingController(text: '2000-01-01');
  String _holdingPeriod = '1 week';
  bool _loading = false;
  String? _error;
  Map<String, dynamic>? _result;

  @override
  void dispose() {
    _tickersController.dispose();
    _startController.dispose();
    super.dispose();
  }

  List<String> get _tickers => _tickersController.text
      .split(RegExp(r'[,\s]+'))
      .map((value) => value.trim().toUpperCase())
      .where((value) => value.isNotEmpty)
      .toSet()
      .toList();

  Future<void> _run() async {
    FocusScope.of(context).unfocus();
    setState(() {
      _loading = true;
      _error = null;
      _result = null;
    });

    try {
      final result = await PortfolioApiClient(
        baseUrl: widget.apiUrl(),
        identityToken: widget.identityToken(),
      ).recommendation({
        'tickers': _tickers,
        'start_date': _startController.text.trim(),
        'holding_period': _holdingPeriod,
        'max_long_weight': 0.35,
        'turnover_penalty_bps': 25.0,
      });
      if (mounted) {
        setState(() => _result = result);
      }
    } on ApiException catch (error) {
      if (mounted) {
        setState(() => _error = error.message);
      }
    } catch (error) {
      if (mounted) {
        setState(() => _error = 'Unexpected error: $error');
      }
    } finally {
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(12, 4, 12, 24),
      children: [
        Text(
          'Latest recommendation',
          style: Theme.of(context).textTheme.headlineSmall,
        ),
        const SizedBox(height: 4),
        const Text('Run the turnover-controlled diffusion portfolio model.'),
        const SizedBox(height: 16),
        TextField(
          controller: _tickersController,
          autocorrect: false,
          textCapitalization: TextCapitalization.characters,
          decoration: const InputDecoration(
            labelText: 'Ticker symbols',
            helperText: 'Separate symbols with commas',
            border: OutlineInputBorder(),
          ),
        ),
        const SizedBox(height: 12),
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(
              child: DropdownButtonFormField<String>(
                initialValue: _holdingPeriod,
                decoration: const InputDecoration(
                  labelText: 'Holding period',
                  border: OutlineInputBorder(),
                ),
                items: _holdingPeriods
                    .map(
                      (value) => DropdownMenuItem(
                        value: value,
                        child: Text(value),
                      ),
                    )
                    .toList(),
                onChanged: (value) {
                  if (value != null) {
                    setState(() => _holdingPeriod = value);
                  }
                },
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: TextField(
                controller: _startController,
                keyboardType: TextInputType.datetime,
                decoration: const InputDecoration(
                  labelText: 'History starts',
                  hintText: 'YYYY-MM-DD',
                  border: OutlineInputBorder(),
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 14),
        FilledButton.icon(
          onPressed: _loading ? null : _run,
          icon: _loading
              ? const SizedBox.square(
                  dimension: 18,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Icon(Icons.auto_graph),
          label: Text(_loading ? 'Running in Google Cloud…' : 'Suggest weights'),
        ),
        if (_error != null) _ErrorCard(message: _error!),
        if (_result != null) _RecommendationResult(result: _result!),
      ],
    );
  }
}

class _RecommendationResult extends StatelessWidget {
  const _RecommendationResult({required this.result});

  final Map<String, dynamic> result;

  @override
  Widget build(BuildContext context) {
    final data = _map(result['data']);
    final model = _map(result['model']);
    final diagnostics = _map(result['diagnostics']);
    final weights = _listOfMaps(result['weights']);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SizedBox(height: 14),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Recommended allocation',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const SizedBox(height: 4),
                Text(
                  'Data through ${data['data_through'] ?? '—'} • '
                  '${data['observations'] ?? '—'} observations',
                ),
                const Divider(height: 24),
                for (final row in weights)
                  _WeightRow(
                    asset: '${row['asset'] ?? ''}',
                    weight: _number(row['recommended']),
                  ),
              ],
            ),
          ),
        ),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Diagnostics', style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 10),
                _MetricRow('Selected diffusion T', _decimal(model['selected_t'])),
                _MetricRow('Turnover', _percent(model['turnover'])),
                _MetricRow(
                  'Expected return / period',
                  _percent(diagnostics['expected_return_per_period']),
                ),
                _MetricRow(
                  'Volatility / period',
                  _percent(diagnostics['volatility_per_period']),
                ),
                _MetricRow(
                  'Sharpe / period',
                  _decimal(diagnostics['sharpe_per_period']),
                ),
              ],
            ),
          ),
        ),
        const Padding(
          padding: EdgeInsets.all(8),
          child: Text(
            'Research output only; not individualized investment advice.',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 12),
          ),
        ),
      ],
    );
  }
}

class BacktestPage extends StatefulWidget {
  const BacktestPage({
    required this.apiUrl,
    required this.identityToken,
    super.key,
  });

  final String Function() apiUrl;
  final String Function() identityToken;

  @override
  State<BacktestPage> createState() => _BacktestPageState();
}

class _BacktestPageState extends State<BacktestPage> {
  final _tickersController =
      TextEditingController(text: 'AAPL, MSFT, NVDA, GOOGL, AMZN');
  String _holdingPeriod = '1 month';
  bool _loading = false;
  String? _error;
  Map<String, dynamic>? _result;

  @override
  void dispose() {
    _tickersController.dispose();
    super.dispose();
  }

  Future<void> _run() async {
    FocusScope.of(context).unfocus();
    final tickers = _tickersController.text
        .split(RegExp(r'[,\s]+'))
        .map((value) => value.trim().toUpperCase())
        .where((value) => value.isNotEmpty)
        .toSet()
        .toList();

    setState(() {
      _loading = true;
      _error = null;
      _result = null;
    });

    try {
      final result = await PortfolioApiClient(
        baseUrl: widget.apiUrl(),
        identityToken: widget.identityToken(),
      ).backtest({
        'tickers': tickers,
        'start_date': '2000-01-01',
        'holding_period': _holdingPeriod,
        'strategy': 'Turnover-Controlled Exact Diffusion',
        'oos_start': '2019-01-01',
        'transaction_cost_bps': 25.0,
        'initial_capital': 10000.0,
        'max_long_weight': 0.35,
      });
      if (mounted) {
        setState(() => _result = result);
      }
    } on ApiException catch (error) {
      if (mounted) {
        setState(() => _error = error.message);
      }
    } catch (error) {
      if (mounted) {
        setState(() => _error = 'Unexpected error: $error');
      }
    } finally {
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(12, 4, 12, 24),
      children: [
        Text('Historical backtest', style: Theme.of(context).textTheme.headlineSmall),
        const SizedBox(height: 4),
        const Text('Evaluate the strategy after transaction costs.'),
        const SizedBox(height: 16),
        TextField(
          controller: _tickersController,
          autocorrect: false,
          textCapitalization: TextCapitalization.characters,
          decoration: const InputDecoration(
            labelText: 'Ticker symbols',
            border: OutlineInputBorder(),
          ),
        ),
        const SizedBox(height: 12),
        DropdownButtonFormField<String>(
          initialValue: _holdingPeriod,
          decoration: const InputDecoration(
            labelText: 'Holding period',
            border: OutlineInputBorder(),
          ),
          items: _holdingPeriods
              .map(
                (value) => DropdownMenuItem(value: value, child: Text(value)),
              )
              .toList(),
          onChanged: (value) {
            if (value != null) {
              setState(() => _holdingPeriod = value);
            }
          },
        ),
        const SizedBox(height: 14),
        FilledButton.icon(
          onPressed: _loading ? null : _run,
          icon: _loading
              ? const SizedBox.square(
                  dimension: 18,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Icon(Icons.play_arrow),
          label: Text(_loading ? 'Running backtest…' : 'Run backtest'),
        ),
        if (_loading)
          const Padding(
            padding: EdgeInsets.only(top: 8),
            child: Text(
              'Keep the app open. A long historical calculation can take several minutes.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 12),
            ),
          ),
        if (_error != null) _ErrorCard(message: _error!),
        if (_result != null) _BacktestResult(result: _result!),
      ],
    );
  }
}

class _BacktestResult extends StatelessWidget {
  const _BacktestResult({required this.result});

  final Map<String, dynamic> result;

  @override
  Widget build(BuildContext context) {
    final initial = _number(result['initial_capital']);
    final net = _number(result['net_final_value']);
    final gross = _number(result['gross_final_value']);
    final returnPercent = initial == 0 ? 0.0 : net / initial - 1.0;
    final metrics = _map(result['net_metrics']);

    return Card(
      margin: const EdgeInsets.only(top: 14),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Backtest result', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 12),
            _MetricRow('Starting value', _money(initial)),
            _MetricRow('Final value after costs', _money(net)),
            _MetricRow('Final value before costs', _money(gross)),
            _MetricRow('Total return after costs', _percent(returnPercent)),
            _MetricRow('Annualized return', _percent(metrics['CAGR'])),
            _MetricRow('Annualized volatility', _percent(metrics['Annualized vol'])),
            _MetricRow('Maximum drawdown', _percent(metrics['Max drawdown'])),
            const Divider(height: 24),
            const Text(
              'Historical research simulation; past performance does not guarantee future results.',
              style: TextStyle(fontSize: 12),
            ),
          ],
        ),
      ),
    );
  }
}

class _WeightRow extends StatelessWidget {
  const _WeightRow({required this.asset, required this.weight});

  final String asset;
  final double weight;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        children: [
          Expanded(child: Text(asset, style: const TextStyle(fontWeight: FontWeight.w600))),
          Text(_percent(weight), style: Theme.of(context).textTheme.titleMedium),
        ],
      ),
    );
  }
}

class _MetricRow extends StatelessWidget {
  const _MetricRow(this.label, this.value);

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Expanded(child: Text(label)),
          Text(value, style: const TextStyle(fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}

class _ErrorCard extends StatelessWidget {
  const _ErrorCard({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Theme.of(context).colorScheme.errorContainer,
      margin: const EdgeInsets.only(top: 12),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(Icons.error_outline, color: Theme.of(context).colorScheme.error),
            const SizedBox(width: 10),
            Expanded(child: Text(message)),
          ],
        ),
      ),
    );
  }
}

const _holdingPeriods = <String>[
  '1 week',
  '2 weeks',
  '1 month',
  '2 months',
  '3 months',
];

Map<String, dynamic> _map(dynamic value) =>
    value is Map<String, dynamic> ? value : <String, dynamic>{};

List<Map<String, dynamic>> _listOfMaps(dynamic value) => value is List
    ? value.whereType<Map<String, dynamic>>().toList()
    : <Map<String, dynamic>>[];

double _number(dynamic value) => value is num ? value.toDouble() : 0.0;

String _percent(dynamic value) => '${(_number(value) * 100).toStringAsFixed(2)}%';

String _decimal(dynamic value) =>
    value is num ? value.toDouble().toStringAsFixed(3) : '—';

String _money(double value) => '\$${value.toStringAsFixed(2)}';
