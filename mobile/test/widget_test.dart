import 'package:diffusion_portfolio_mobile/main.dart';
import 'package:diffusion_portfolio_mobile/diffusion_ols_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('shows Google connection and portfolio controls', (tester) async {
    // Google authentication itself is exercised on an Android device.
    await tester.pumpWidget(
      const DiffusionPortfolioApp(initializeGoogleSignIn: false),
    );

    expect(find.text('Secure Google account'), findsOneWidget);
    expect(find.text('API URL'), findsOneWidget);
    expect(find.textContaining('OAuth client ID'), findsNothing);
    await tester.tap(find.text('Done'));
    await tester.pumpAndSettle();

    expect(find.text('Latest recommendation'), findsOneWidget);
    expect(find.text('Suggest weights'), findsOneWidget);

    await tester.tap(find.text('Backtest'));
    await tester.pumpAndSettle();

    expect(find.text('Historical backtest'), findsOneWidget);
    expect(find.text('Data history starts (YYYY-MM-DD)'), findsOneWidget);
    expect(find.text('Holding period'), findsOneWidget);
    expect(find.text('Estimation lookback observations'), findsOneWidget);

    await tester.tap(find.text('Diffusion OLS'));
    await tester.pumpAndSettle();
    expect(find.text('Diffusion OLS: Market Exposure'), findsOneWidget);
    final olsScroll = find.descendant(
      of: find.byType(DiffusionOlsPage),
      matching: find.byType(Scrollable),
    ).first;
    await tester.scrollUntilVisible(
      find.text('Backtest start date (YYYY-MM-DD)'),
      400,
      scrollable: olsScroll,
    );
    await tester.pumpAndSettle();
    expect(find.text('Run Diffusion OLS exposure analysis'), findsOneWidget);
    expect(find.text('Backtest start date (YYYY-MM-DD)'), findsOneWidget);
    await tester.tap(find.text('Feasible Tuning').last);
    await tester.pumpAndSettle();
    expect(find.textContaining('Run automatically selects c'), findsOneWidget);
    expect(find.text('Enter stocks / download from Yahoo'), findsOneWidget);
    expect(find.text('Tickers (Yahoo Finance)'), findsOneWidget);
    expect(find.widgetWithText(TextField, 'Tickers (Yahoo Finance)'), findsOneWidget);
    expect(find.text('Maximum a'), findsNothing);
    expect(find.text('Trace tuning versus old Portfolio'), findsNothing);
  });
  testWidgets('all navigation tabs fit on a narrow iPhone', (tester) async {
    tester.view.physicalSize = const Size(320, 568);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    await tester.pumpWidget(
      const DiffusionPortfolioApp(initializeGoogleSignIn: false),
    );
    await tester.tap(find.text('Done'));
    await tester.pumpAndSettle();
    final navigation = find.byType(NavigationBar);
    expect(find.descendant(of: navigation, matching: find.text('Tuning')), findsOneWidget);
    for (final label in ['Portfolio', 'Backtest', 'OLS', 'Tuning']) {
      final tab = find.descendant(of: navigation, matching: find.text(label));
      final rect = tester.getRect(tab);
      expect(rect.left, greaterThanOrEqualTo(0));
      expect(rect.right, lessThanOrEqualTo(320));
    }
    await tester.tap(find.descendant(of: navigation, matching: find.text('Tuning')));
    await tester.pumpAndSettle();
    expect(find.textContaining('Run automatically selects c'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

}
