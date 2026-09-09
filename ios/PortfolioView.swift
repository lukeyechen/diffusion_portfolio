import SwiftUI

struct PortfolioView: View {
    private let holdingPeriods = ["1 week", "2 weeks", "1 month", "2 months", "3 months"]
    private let apiClient = APIClient()

    @State private var tickersText = "AAPL,MSFT,NVDA,GOOGL,AMZN"
    @State private var holdingPeriod = "1 week"
    @State private var gamma = 3.0
    @State private var maxWeightPct = 40.0
    @State private var isLoading = false
    @State private var result: PortfolioRecommendationResponse?
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            Form {
                Section("Portfolio Inputs") {
                    TextField("Tickers", text: $tickersText)
                        .textInputAutocapitalization(.characters)
                        .autocorrectionDisabled()

                    Picker("Holding period", selection: $holdingPeriod) {
                        ForEach(holdingPeriods, id: \.self) { value in
                            Text(value).tag(value)
                        }
                    }

                    HStack {
                        Text("Risk aversion")
                        Spacer()
                        TextField("3.0", value: $gamma, format: .number.precision(.fractionLength(1)))
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 80)
                    }

                    HStack {
                        Text("Max asset weight")
                        Spacer()
                        TextField("40", value: $maxWeightPct, format: .number.precision(.fractionLength(0)))
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 60)
                        Text("%")
                    }
                }

                Section {
                    Button {
                        Task { await calculate() }
                    } label: {
                        HStack {
                            Spacer()
                            if isLoading {
                                ProgressView()
                            } else {
                                Label("Calculate Portfolio", systemImage: "chart.pie.fill")
                            }
                            Spacer()
                        }
                    }
                    .disabled(isLoading)
                }

                if let errorMessage {
                    Section("Error") {
                        Text(errorMessage)
                            .foregroundStyle(.red)
                    }
                }

                if let result {
                    Section("Recommended Portfolio") {
                        ForEach(result.weights.sorted { $0.recommendedWeight > $1.recommendedWeight }) { row in
                            HStack {
                                Text(row.ticker)
                                    .fontWeight(.semibold)
                                Spacer()
                                Text(row.recommendedWeight, format: .percent.precision(.fractionLength(2)))
                                    .monospacedDigit()
                            }
                        }
                    }

                    Section("Model Details") {
                        detailRow("Selected T", String(format: "%.2f", result.selectedT))
                        detailRow("Turnover", result.turnover.formatted(.percent.precision(.fractionLength(2))))
                        detailRow("TC penalty", String(format: "%.0f bp", result.turnoverPenaltyBps))
                        detailRow("Rebalance step", String(format: "%.0f%%", result.rebalanceStepPct))
                        detailRow("Lookback", "\(result.lookback) periods")
                        detailRow("Data through", result.dataThrough)
                    }

                    Section("Weight Path") {
                        ForEach(result.weights) { row in
                            VStack(alignment: .leading, spacing: 6) {
                                Text(row.ticker).fontWeight(.semibold)
                                HStack {
                                    Text("Previous")
                                    Spacer()
                                    Text(row.previousDrifted, format: .percent.precision(.fractionLength(2)))
                                }
                                HStack {
                                    Text("Raw TC target")
                                    Spacer()
                                    Text(row.rawTarget, format: .percent.precision(.fractionLength(2)))
                                }
                                HStack {
                                    Text("Recommended")
                                    Spacer()
                                    Text(row.recommendedWeight, format: .percent.precision(.fractionLength(2)))
                                        .fontWeight(.semibold)
                                }
                            }
                            .padding(.vertical, 4)
                        }
                    }
                }
            }
            .navigationTitle("Diffusion Portfolio")
        }
    }

    @ViewBuilder
    private func detailRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value)
                .foregroundStyle(.secondary)
                .monospacedDigit()
        }
    }

    @MainActor
    private func calculate() async {
        errorMessage = nil
        result = nil
        isLoading = true
        defer { isLoading = false }

        let tickers = tickersText
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).uppercased() }
            .filter { !$0.isEmpty }

        guard !tickers.isEmpty else {
            errorMessage = "Enter at least one ticker."
            return
        }

        let request = PortfolioRecommendationRequest(
            tickers: tickers,
            startDate: "2000-01-01",
            holdingPeriod: holdingPeriod,
            gamma: gamma,
            maxWeight: maxWeightPct / 100.0,
            syntheticEquivalentM: 500,
            beta: 1.0,
            reverseSteps: 100,
            turnoverPenaltyBps: 25.0,
            rebalanceStepPct: nil,
            replayStart: "2019-01-01"
        )

        do {
            result = try await apiClient.portfolioRecommendation(request)
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

#Preview {
    PortfolioView()
}
