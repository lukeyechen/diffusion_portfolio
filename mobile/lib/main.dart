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
  final _startDateController = TextEditingController(text: '2000-01-01');
  final _lookbackController = TextEditingController(text: '520');
  final _gammaController = TextEditingController(text: '3.00');
  final _maxWeightController = TextEditingController(text: '0.40');
  final _capitalController = TextEditingController(text: '10000.00');
  final _syntheticMController = TextEditingController(text: '500');
  final _betaController = TextEditingController(text: '1.00');
  final _reverseStepsController = TextEditingController(text: '100');
  final _tradingCostController = TextEditingController(text: '25.00');
  final _turnoverPenaltyController = TextEditingController(text: '25.00');
  final _rebalanceController = TextEditingController(text: '50.00');
  final _oosStartController = TextEditingController(text: '2019-01-01');
  String _holdingPeriod = '1 week';
  String _strategy = 'Turnover-Controlled Exact Diffusion';
  String _turnoverMode = 'Validated preset';
  bool _loading = false;
  String? _error;
  Map<String, dynamic>? _result;

  @override
  void dispose() {
    _tickersController.dispose();
    _startDateController.dispose();
    _lookbackController.dispose();
    _gammaController.dispose();
    _maxWeightController.dispose();
    _capitalController.dispose();
    _syntheticMController.dispose();
    _betaController.dispose();
    _reverseStepsController.dispose();
    _tradingCostController.dispose();
    _turnoverPenaltyController.dispose();
    _rebalanceController.dispose();
    _oosStartController.dispose();
    super.dispose();
  }

  void _setHoldingPeriod(String value) {
    setState(() {
      _holdingPeriod = value;
      _lookbackController.text = '${_defaultLookbacks[value] ?? 120}';
      if (_turnoverMode == 'Validated preset') {
        _turnoverPenaltyController.text = '25.00';
        _rebalanceController.text = value == '1 week' ? '50.00' : '100.00';
      }
    });
  }

  void _setTurnoverMode(String value) {
    setState(() {
      _turnoverMode = value;
      if (value == 'Validated preset') {
        _turnoverPenaltyController.text = '25.00';
        _rebalanceController.text =
            _holdingPeriod == '1 week' ? '50.00' : '100.00';
      }
    });
  }

  Widget _settingField(
    TextEditingController controller,
    String label, {
    bool enabled = true,
    bool integer = false,
    bool date = false,
  }) {
    return Padding(
      padding: const EdgeInsets.only(top: 12),
      child: TextField(
        controller: controller,
        enabled: enabled,
        keyboardType: date
            ? TextInputType.datetime
            : TextInputType.numberWithOptions(decimal: !integer),
        decoration: InputDecoration(
          labelText: label,
          border: const OutlineInputBorder(),
        ),
      ),
    );
  }

  Future<void> _run() async {
    FocusScope.of(context).unfocus();
    final tickers = _tickersController.text
        .split(RegExp(r'[,\s]+'))
        .map((value) => value.trim().toUpperCase())
        .where((value) => value.isNotEmpty)
        .toSet()
        .toList();

    final lookback = int.tryParse(_lookbackController.text.trim());
    final gamma = double.tryParse(_gammaController.text.trim());
    final maxWeight = double.tryParse(_maxWeightController.text.trim());
    final capital = double.tryParse(_capitalController.text.trim());
    final syntheticM = int.tryParse(_syntheticMController.text.trim());
    final beta = double.tryParse(_betaController.text.trim());
    final reverseSteps = int.tryParse(_reverseStepsController.text.trim());
    final tradingCost = double.tryParse(_tradingCostController.text.trim());
    final turnoverPenalty =
        double.tryParse(_turnoverPenaltyController.text.trim());
    final rebalance = double.tryParse(_rebalanceController.text.trim());

    if (tickers.isEmpty ||
        lookback == null ||
        gamma == null ||
        maxWeight == null ||
        capital == null ||
        syntheticM == null ||
        beta == null ||
        reverseSteps == null ||
        tradingCost == null ||
        turnoverPenalty == null ||
        rebalance == null) {
      setState(() => _error = 'Enter valid values in every backtest setting.');
      return;
    }

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
        'start_date': _startDateController.text.trim(),
        'holding_period': _holdingPeriod,
        'lookback': lookback,
        'strategy': _strategy,
        'gamma': gamma,
        'synthetic_equivalent_m': syntheticM,
        'beta': beta,
        'reverse_steps': reverseSteps,
        'turnover_penalty_bps': turnoverPenalty,
        'rebalance_percent': rebalance,
        'max_long_weight': maxWeight,
        'oos_start': _oosStartController.text.trim(),
        'transaction_cost_bps': tradingCost,
        'initial_capital': capital,
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
        _settingField(
          _startDateController,
          'Data history starts (YYYY-MM-DD)',
          date: true,
        ),
        const SizedBox(height: 12),
        DropdownButtonFormField<String>(
          initialValue: _holdingPeriod,
          isExpanded: true,
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
              _setHoldingPeriod(value);
            }
          },
        ),
        _settingField(
          _lookbackController,
          'Estimation lookback observations',
          integer: true,
        ),
        const SizedBox(height: 12),
        DropdownButtonFormField<String>(
          initialValue: _strategy,
          isExpanded: true,
          decoration: const InputDecoration(
            labelText: 'Strategy to evaluate',
            border: OutlineInputBorder(),
          ),
          items: _backtestStrategies
              .map(
                (value) => DropdownMenuItem(value: value, child: Text(value)),
              )
              .toList(),
          onChanged: (value) {
            if (value != null) {
              setState(() => _strategy = value);
            }
          },
        ),
        _settingField(_gammaController, 'Risk aversion γ'),
        _settingField(_maxWeightController, 'Maximum weight per asset'),
        _settingField(_capitalController, 'Initial capital (\$)'),
        _settingField(
          _syntheticMController,
          'Synthetic-equivalent M',
          integer: true,
        ),
        _settingField(_betaController, 'Constant β'),
        _settingField(
          _reverseStepsController,
          'Reverse SDE steps',
          integer: true,
        ),
        _settingField(
          _tradingCostController,
          'Realized trading cost (bps)',
        ),
        const SizedBox(height: 12),
        DropdownButtonFormField<String>(
          initialValue: _turnoverMode,
          isExpanded: true,
          decoration: const InputDecoration(
            labelText: 'Turnover-control settings',
            border: OutlineInputBorder(),
          ),
          items: const [
            DropdownMenuItem(
              value: 'Validated preset',
              child: Text('Validated preset'),
            ),
            DropdownMenuItem(value: 'Custom', child: Text('Custom')),
          ],
          onChanged: (value) {
            if (value != null) {
              _setTurnoverMode(value);
            }
          },
        ),
        _settingField(
          _turnoverPenaltyController,
          'TC optimizer penalty (bps)',
          enabled: _turnoverMode == 'Custom',
        ),
        _settingField(
          _rebalanceController,
          'Rebalance step (%)',
          enabled: _turnoverMode == 'Custom',
        ),
        _settingField(
          _oosStartController,
          'True OOS evaluation start (YYYY-MM-DD)',
          date: true,
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
    final grossMetrics = _map(result['gross_metrics']);
    final netMetrics = _map(result['net_metrics']);
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
    final calendarYears = _listOfMaps(result['calendar_year_returns']);
    final periods = _listOfMaps(result['periods']);
    final latestRows = _listOfMaps(result['latest_weights']);
    final latestWeights = latestRows.firstWhere(
      (row) => '${row['Method'] ?? ''}' == strategy,
      orElse: () => <String, dynamic>{},
    );

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
            _MetricRow('Annualized return', _percent(netMetrics['CAGR'])),
            _MetricRow(
              'Annualized volatility',
              _percent(netMetrics['Annualized vol']),
            ),
            _MetricRow('Net Sharpe', _decimal(netMetrics['Sharpe'])),
            _MetricRow(
              'Maximum drawdown',
              _percent(netMetrics['Max drawdown']),
            ),
            _MetricRow('Average turnover', _percent(summary['Average turnover'])),
            const Divider(height: 24),
            Text(
              'Gross and net performance',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 6),
            _PerformanceTable(gross: grossMetrics, net: netMetrics),
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
            Text(
              'Calendar-year realized return',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 6),
            _CalendarReturnTable(rows: calendarYears),
            const Divider(height: 24),
            Text(
              'Latest backtested target weights',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 6),
            ...latestWeights.entries
                .where((entry) => entry.key != 'Method')
                .map(
                  (entry) => _MetricRow(entry.key, _percent(entry.value)),
                ),
            const Divider(height: 24),
            ExpansionTile(
              tilePadding: EdgeInsets.zero,
              childrenPadding: EdgeInsets.zero,
              title: const Text('Rolling OOS detail'),
              children: [
                _RollingDetailTable(rows: periods),
              ],
            ),
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

class _PerformanceTable extends StatelessWidget {
  const _PerformanceTable({required this.gross, required this.net});

  final Map<String, dynamic> gross;
  final Map<String, dynamic> net;

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: DataTable(
        columnSpacing: 22,
        columns: const [
          DataColumn(label: Text('Metric')),
          DataColumn(label: Text('Gross'), numeric: true),
          DataColumn(label: Text('Net'), numeric: true),
        ],
        rows: [
          _performanceRow('Total return', 'Total return', gross, net),
          _performanceRow('CAGR', 'CAGR', gross, net),
          _performanceRow(
            'Annualized volatility',
            'Annualized vol',
            gross,
            net,
          ),
          _performanceRow('Sharpe', 'Sharpe', gross, net, decimal: true),
          _performanceRow('Realized CER', 'Realized CER', gross, net),
          _performanceRow('Max drawdown', 'Max drawdown', gross, net),
          _performanceRow('Positive periods', 'Positive periods', gross, net),
        ],
      ),
    );
  }
}

DataRow _performanceRow(
  String label,
  String key,
  Map<String, dynamic> gross,
  Map<String, dynamic> net, {
  bool decimal = false,
}) {
  final formatter = decimal ? _decimal : _percent3;
  return DataRow(
    cells: [
      DataCell(Text(label)),
      DataCell(Text(formatter(gross[key]))),
      DataCell(Text(formatter(net[key]))),
    ],
  );
}

class _CalendarReturnTable extends StatelessWidget {
  const _CalendarReturnTable({required this.rows});

  final List<Map<String, dynamic>> rows;

  @override
  Widget build(BuildContext context) {
    if (rows.isEmpty) {
      return const Text('No calendar-year returns available.');
    }
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: DataTable(
        columnSpacing: 32,
        columns: const [
          DataColumn(label: Text('Year')),
          DataColumn(label: Text('Net return'), numeric: true),
        ],
        rows: rows
            .map(
              (row) => DataRow(
                cells: [
                  DataCell(Text(_integer(row['year']))),
                  DataCell(Text(_percent(row['net_return']))),
                ],
              ),
            )
            .toList(),
      ),
    );
  }
}

class _RollingDetailTable extends StatelessWidget {
  const _RollingDetailTable({required this.rows});

  final List<Map<String, dynamic>> rows;

  @override
  Widget build(BuildContext context) {
    if (rows.isEmpty) {
      return const Text('No rolling OOS details available.');
    }
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: DataTable(
        columnSpacing: 18,
        columns: const [
          DataColumn(label: Text('Date')),
          DataColumn(label: Text('Gross return'), numeric: true),
          DataColumn(label: Text('Turnover'), numeric: true),
          DataColumn(label: Text('Trading cost'), numeric: true),
          DataColumn(label: Text('Net return'), numeric: true),
          DataColumn(label: Text('Selected T'), numeric: true),
          DataColumn(label: Text('Year'), numeric: true),
        ],
        rows: rows
            .map(
              (row) => DataRow(
                cells: [
                  DataCell(Text(_dateOnly(row['date']))),
                  DataCell(Text(_percent3(row['gross_return']))),
                  DataCell(Text(_percent(row['turnover']))),
                  DataCell(Text(_percent3(row['trading_cost']))),
                  DataCell(Text(_percent3(row['net_return']))),
                  DataCell(Text(_decimal(row['selected_t']))),
                  DataCell(Text(_integer(row['year']))),
                ],
              ),
            )
            .toList(),
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

const _defaultLookbacks = <String, int>{
  '1 week': 520,
  '2 weeks': 260,
  '1 month': 120,
  '2 months': 60,
  '3 months': 40,
};

const _backtestStrategies = <String>[
  'Equal Weight',
  'Classical MV',
  'Classical MV + LW',
  'Exact Diffusion (Best-T)',
  '50% Exact Diff + 50% EW',
  'Turnover-Controlled Exact Diffusion',
];

Map<String, dynamic> _map(dynamic value) =>
    value is Map<String, dynamic> ? value : <String, dynamic>{};

List<Map<String, dynamic>> _listOfMaps(dynamic value) => value is List
    ? value.whereType<Map<String, dynamic>>().toList()
    : <Map<String, dynamic>>[];

double _number(dynamic value) => value is num ? value.toDouble() : 0.0;

String _percent(dynamic value) => '${(_number(value) * 100).toStringAsFixed(2)}%';

String _percent3(dynamic value) =>
    '${(_number(value) * 100).toStringAsFixed(3)}%';

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
