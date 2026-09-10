import 'package:diffusion_portfolio_mobile/main.dart';
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
  });
}
