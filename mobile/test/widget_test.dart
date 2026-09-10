import 'package:diffusion_portfolio_mobile/main.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('shows private connection and portfolio controls', (tester) async {
    await tester.pumpWidget(const DiffusionPortfolioApp());

    expect(find.text('Private Google Cloud connection'), findsOneWidget);
    expect(find.text('Latest recommendation'), findsOneWidget);
    expect(find.text('Suggest weights'), findsOneWidget);
  });
}
