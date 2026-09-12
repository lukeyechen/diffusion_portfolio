import 'dart:async';

import 'package:flutter/material.dart';

import 'api_client.dart';
import 'google_auth_controller.dart';
import 'google_sign_in_button.dart';

void main() {
  runApp(const DiffusionPortfolioApp());
}

class DiffusionPortfolioApp extends StatelessWidget {
  const DiffusionPortfolioApp({
    super.key,
    this.initializeGoogleSignIn = true,
  });

  final bool initializeGoogleSignIn;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Diffusion Portfolio',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xff3157d5)),
        useMaterial3: true,
      ),
      home: PortfolioHomePage(
        initializeGoogleSignIn: initializeGoogleSignIn,
      ),
    );
  }
}

class PortfolioHomePage extends StatefulWidget {
  const PortfolioHomePage({
    super.key,
    this.initializeGoogleSignIn = true,
  });

  final bool initializeGoogleSignIn;

  @override
  State<PortfolioHomePage> createState() => _PortfolioHomePageState();
}

class _PortfolioHomePageState extends State<PortfolioHomePage> {
  static const _configuredApiUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue:
        'https://diffusion-portfolio-api-2n4i7hamea-ue.a.run.app',
  );

  final _apiUrlController = TextEditingController(text: _configuredApiUrl);
  final _auth = GoogleAuthController();
  int _selectedPage = 0;
  bool _showConnection = true;

  @override
  void initState() {
    super.initState();
    _auth.addListener(_authChanged);
    if (widget.initializeGoogleSignIn) {
      unawaited(_auth.initialize());
    }
  }

  void _authChanged() {
    if (mounted) {
      setState(() {});
    }
  }

  @override
  void dispose() {
    _apiUrlController.dispose();
    _auth.removeListener(_authChanged);
    _auth.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Diffusion Portfolio'),
        actions: [
          IconButton(
            tooltip: 'Google account and API connection',
            onPressed: () => setState(() => _showConnection = !_showConnection),
            icon: Icon(_auth.isSignedIn ? Icons.verified_user : Icons.login),
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            if (_showConnection)
              _GoogleConnectionCard(
                apiUrlController: _apiUrlController,
                auth: _auth,
                onDone: () => setState(() => _showConnection = false),
              ),
            Expanded(
              child: IndexedStack(
                index: _selectedPage,
                children: [
                  RecommendationPage(
                    apiUrl: () => _apiUrlController.text,
                    identityToken: () => _auth.idToken ?? '',
                  ),
                  BacktestPage(
                    apiUrl: () => _apiUrlController.text,
                    identityToken: () => _auth.idToken ?? '',
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

class _GoogleConnectionCard extends StatelessWidget {
  const _GoogleConnectionCard({
    required this.apiUrlController,
    required this.auth,
    required this.onDone,
  });

  final TextEditingController apiUrlController;
  final GoogleAuthController auth;
  final VoidCallback onDone;

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
                  'Secure Google account',
                  style: TextStyle(fontWeight: FontWeight.w600),
                ),
              ],
            ),
            const SizedBox(height: 12),
            TextField(
              controller: apiUrlController,
              keyboardType: TextInputType.url,
              autocorrect: false,
              decoration: const InputDecoration(
                labelText: 'API URL',
                hintText: 'https://diffusion-portfolio-api-…run.app',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            if (!auth.isConfigured)
              const _AuthenticationNotice(
                icon: Icons.settings_outlined,
                message:
                    'Google sign-in needs the project OAuth client ID before it can be used.',
              )
            else if (auth.isSignedIn)
              _SignedInAccount(auth: auth)
            else if (auth.isBusy)
              const Center(child: CircularProgressIndicator())
            else
              buildGoogleSignInButton(
                onPressed: auth.signIn,
              ),
            if (auth.error != null) ...[
              const SizedBox(height: 8),
              Text(
                auth.error!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            ],
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerRight,
              child: FilledButton.tonal(
                onPressed: onDone,
                child: const Text('Done'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _AuthenticationNotice extends StatelessWidget {
  const _AuthenticationNotice({required this.icon, required this.message});

  final IconData icon;
  final String message;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 20),
        const SizedBox(width: 8),
        Expanded(child: Text(message)),
      ],
    );
  }
}

class _SignedInAccount extends StatelessWidget {
  const _SignedInAccount({required this.auth});

  final GoogleAuthController auth;

  @override
  Widget build(BuildContext context) {
    final user = auth.user!;
    return Row(
      children: [
        const CircleAvatar(child: Icon(Icons.person)),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                user.displayName ?? 'Google account',
                style: const TextStyle(fontWeight: FontWeight.w600),
              ),
              Text(user.email, style: const TextStyle(fontSize: 12)),
            ],
          ),
        ),
        TextButton(
          onPressed: auth.isBusy ? null : auth.signOut,
          child: const Text('Sign out'),
        ),
      ],
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
    final strategy = '${result['strategy'] ?? ''}';
    final summaries = _listOfMaps(result['all_method_summary']);
    final summary = summaries.firstWhere(
      (row) => '${row['Method'] ?? ''}' == strategy,
      orElse: () => <String, dynamic>{},
    );
    final startDate = _dateOnly(summary['OOS start']);
    final endDate = _dateOnly(summary['OOS end']);
    final periodCount = summary['OOS periods'] is num
        ? (summary['OOS periods'] as num).toInt().toString()
        : '—';
    final holdingPeriod = '${result['holding_period'] ?? '—'}';
    final tSelection = _map(result['t_selection']);
    final tFrequency = _listOfMaps(tSelection['frequency']);

    return Card(
      margin: const EdgeInsets.only(top: 14),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Backtest result', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 6),
            const Text(
              'These values cover the complete historical test, not one holding period.',
              style: TextStyle(fontSize: 12),
            ),
            const SizedBox(height: 12),
            _MetricRow('Backtest dates', '$startDate to $endDate'),
            _MetricRow('Holding/rebalance interval', holdingPeriod),
            _MetricRow('Number of holding periods', periodCount),
            const Divider(height: 24),
            _MetricRow('Starting value', _money(initial)),
            _MetricRow('Final value after costs (full test)', _money(net)),
            _MetricRow('Final value before costs (full test)', _money(gross)),
            _MetricRow('Total return after costs (full test)', _percent(returnPercent)),
            _MetricRow('Annualized return', _percent(metrics['CAGR'])),
            _MetricRow('Annualized volatility', _percent(metrics['Annualized vol'])),
            _MetricRow('Net Sharpe', _decimal(metrics['Sharpe'])),
            _MetricRow('Maximum drawdown', _percent(metrics['Max drawdown'])),
            _MetricRow('Average turnover', _percent(summary['Average turnover'])),
            const Divider(height: 24),
            Text(
              'Selected diffusion horizon T',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 6),
            _MetricRow('Latest selected T', _decimal(tSelection['latest'])),
            _MetricRow('Most frequent T', _decimal(tSelection['most_frequent'])),
            _MetricRow('Median T', _decimal(tSelection['median'])),
            _MetricRow('Mean T', _decimal(tSelection['mean'])),
            _MetricRow(
              'T > 0 selection share',
              _percent(tSelection['positive_fraction']),
            ),
            if (tFrequency.isNotEmpty) ...[
              const SizedBox(height: 8),
              const Text(
                'Selection frequency',
                style: TextStyle(fontWeight: FontWeight.w600),
              ),
              ...tFrequency.map(
                (row) => _MetricRow(
                  'T = ${_decimal(row['t'])}',
                  '${_integer(row['count'])} periods (${_percent(row['fraction'])})',
                ),
              ),
            ],
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

String _integer(dynamic value) =>
    value is num ? value.toInt().toString() : '—';

String _decimal(dynamic value) =>
    value is num ? value.toDouble().toStringAsFixed(3) : '—';

String _dateOnly(dynamic value) {
  final text = value?.toString() ?? '';
  if (text.isEmpty) {
    return '—';
  }
  final timeSeparator = text.indexOf('T');
  return timeSeparator > 0 ? text.substring(0, timeSeparator) : text;
}

String _money(double value) => '\$${value.toStringAsFixed(2)}';
