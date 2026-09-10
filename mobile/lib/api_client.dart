import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

class ApiException implements Exception {
  ApiException(this.message, {this.statusCode});

  final String message;
  final int? statusCode;

  @override
  String toString() => message;
}

class PortfolioApiClient {
  PortfolioApiClient({required this.baseUrl, this.identityToken = ''});

  final String baseUrl;
  final String identityToken;

  Future<Map<String, dynamic>> recommendation(
    Map<String, dynamic> request,
  ) =>
      _post('/v1/portfolio/recommendation', request);

  Future<Map<String, dynamic>> backtest(Map<String, dynamic> request) =>
      _post('/v1/backtest', request);

  Future<Map<String, dynamic>> _post(
    String path,
    Map<String, dynamic> request,
  ) async {
    final normalizedBaseUrl = baseUrl.trim().replaceFirst(RegExp(r'/$'), '');
    if (normalizedBaseUrl.isEmpty) {
      throw ApiException('Enter the private API URL first.');
    }

    final uri = Uri.tryParse('$normalizedBaseUrl$path');
    if (uri == null || !uri.hasScheme || uri.scheme != 'https') {
      throw ApiException('The API URL must be a valid https:// address.');
    }

    final headers = <String, String>{'Content-Type': 'application/json'};
    if (identityToken.trim().isNotEmpty) {
      headers['Authorization'] = 'Bearer ${identityToken.trim()}';
    }

    late http.Response response;
    try {
      response = await http
          .post(uri, headers: headers, body: jsonEncode(request))
          .timeout(const Duration(minutes: 65));
    } on TimeoutException {
      throw ApiException('The cloud calculation timed out after 65 minutes.');
    } on Exception catch (error) {
      throw ApiException('Could not reach the cloud API: $error');
    }

    dynamic decoded;
    try {
      decoded = jsonDecode(response.body);
    } on FormatException {
      throw ApiException(
        'The server returned an unreadable response (${response.statusCode}).',
        statusCode: response.statusCode,
      );
    }

    if (response.statusCode < 200 || response.statusCode >= 300) {
      final detail = decoded is Map<String, dynamic> ? decoded['detail'] : null;
      final message = detail is String
          ? detail
          : response.statusCode == 401 || response.statusCode == 403
              ? 'Access denied. Sign out and sign in with Google again.'
              : 'Cloud request failed (${response.statusCode}).';
      throw ApiException(message, statusCode: response.statusCode);
    }

    if (decoded is! Map<String, dynamic>) {
      throw ApiException('The server response has an unexpected format.');
    }
    return decoded;
  }
}
