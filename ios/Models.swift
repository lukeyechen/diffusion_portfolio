import Foundation

struct PortfolioRecommendationRequest: Encodable {
    let tickers: [String]
    let startDate: String
    let holdingPeriod: String
    let gamma: Double
    let maxWeight: Double
    let syntheticEquivalentM: Int
    let beta: Double
    let reverseSteps: Int
    let turnoverPenaltyBps: Double
    let rebalanceStepPct: Double?
    let replayStart: String

    enum CodingKeys: String, CodingKey {
        case tickers
        case startDate = "start_date"
        case holdingPeriod = "holding_period"
        case gamma
        case maxWeight = "max_weight"
        case syntheticEquivalentM = "synthetic_equivalent_m"
        case beta
        case reverseSteps = "reverse_steps"
        case turnoverPenaltyBps = "turnover_penalty_bps"
        case rebalanceStepPct = "rebalance_step_pct"
        case replayStart = "replay_start"
    }
}

struct PortfolioRecommendationResponse: Decodable {
    let holdingPeriod: String
    let dataThrough: String
    let observations: Int
    let lookback: Int
    let selectedT: Double
    let previousT: Double?
    let turnover: Double
    let turnoverPenaltyBps: Double
    let rebalanceStepPct: Double
    let gamma: Double
    let maxWeight: Double
    let weights: [WeightRow]
    let tValidation: [TValidationRow]

    enum CodingKeys: String, CodingKey {
        case holdingPeriod = "holding_period"
        case dataThrough = "data_through"
        case observations
        case lookback
        case selectedT = "selected_T"
        case previousT = "previous_T"
        case turnover
        case turnoverPenaltyBps = "turnover_penalty_bps"
        case rebalanceStepPct = "rebalance_step_pct"
        case gamma
        case maxWeight = "max_weight"
        case weights
        case tValidation = "t_validation"
    }
}

struct WeightRow: Decodable, Identifiable {
    let ticker: String
    let previousDrifted: Double
    let rawTarget: Double
    let recommendedWeight: Double

    var id: String { ticker }

    enum CodingKeys: String, CodingKey {
        case ticker
        case previousDrifted = "previous_drifted"
        case rawTarget = "raw_target"
        case recommendedWeight = "recommended_weight"
    }
}

struct TValidationRow: Decodable, Identifiable {
    let T: Double
    let meanValidationCER: Double
    let stdValidationCER: Double
    let standardErrorCER: Double?
    let selected: Bool

    var id: Double { T }

    enum CodingKeys: String, CodingKey {
        case T
        case meanValidationCER = "mean_validation_CER"
        case stdValidationCER = "std_validation_CER"
        case standardErrorCER = "standard_error_CER"
        case selected
    }
}

struct APIErrorResponse: Decodable {
    let detail: String
}
