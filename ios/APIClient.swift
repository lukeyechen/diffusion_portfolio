import Foundation

enum APIClientError: LocalizedError {
    case invalidResponse
    case server(String)

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "The server returned an invalid response."
        case .server(let message):
            return message
        }
    }
}

struct APIClient {
    func portfolioRecommendation(
        _ requestBody: PortfolioRecommendationRequest
    ) async throws -> PortfolioRecommendationResponse {
        let url = APIConfig.baseURL.appendingPathComponent("portfolio/recommendation")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 900
        request.httpBody = try JSONEncoder().encode(requestBody)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw APIClientError.invalidResponse
        }

        if (200..<300).contains(http.statusCode) {
            return try JSONDecoder().decode(PortfolioRecommendationResponse.self, from: data)
        }

        if let serverError = try? JSONDecoder().decode(APIErrorResponse.self, from: data) {
            throw APIClientError.server(serverError.detail)
        }
        throw APIClientError.server("Server error (HTTP \(http.statusCode)).")
    }
}
