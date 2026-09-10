import 'package:diffusion_portfolio_mobile/main.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('shows Google connection and portfolio controls', (tester) async {
    await tester.pumpWidget(const DiffusionPortfolioApp());

    expect(find.text('Secure Google account'), findsOneWidget);
    expect(find.textContaining('OAuth client ID'), findsOneWidget);
    await tester.tap(find.text('Done'));
    await tester.pumpAndSettle();

    expect(find.text('Latest recommendation'), findsOneWidget);
    expect(find.text('Suggest weights'), findsOneWidget);
  });
}
