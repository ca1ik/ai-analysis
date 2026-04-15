import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import '../constants/app_constants.dart';

class ApiService {
  final String baseUrl;
  final http.Client _client;

  ApiService({String? baseUrl})
    : baseUrl = baseUrl ?? AppConstants.apiBaseUrl,
      _client = http.Client();

  // ─── Generic GET ─────────────────────────────────────
  Future<Map<String, dynamic>> _get(String endpoint) async {
    try {
      final uri = Uri.parse('$baseUrl$endpoint');
      final response = await _client
          .get(uri)
          .timeout(const Duration(seconds: 10));
      if (response.statusCode == 200) {
        return json.decode(response.body) as Map<String, dynamic>;
      }
      return {'error': 'HTTP ${response.statusCode}'};
    } catch (e) {
      debugPrint('ApiService GET $endpoint: $e');
      return {'error': e.toString()};
    }
  }

  // ─── Generic POST ────────────────────────────────────
  Future<Map<String, dynamic>> _post(
    String endpoint, {
    Map<String, dynamic>? body,
  }) async {
    try {
      final uri = Uri.parse('$baseUrl$endpoint');
      final response = await _client
          .post(
            uri,
            headers: {'Content-Type': 'application/json'},
            body: body != null ? json.encode(body) : null,
          )
          .timeout(const Duration(seconds: 30));
      if (response.statusCode == 200) {
        return json.decode(response.body) as Map<String, dynamic>;
      }
      return {'error': 'HTTP ${response.statusCode}'};
    } catch (e) {
      debugPrint('ApiService POST $endpoint: $e');
      return {'error': e.toString()};
    }
  }

  // ─── Status ──────────────────────────────────────────
  Future<Map<String, dynamic>> getStatus() => _get(AppConstants.statusEndpoint);

  // ─── GPU ─────────────────────────────────────────────
  Future<Map<String, dynamic>> getGpu() => _get(AppConstants.gpuEndpoint);

  // ─── Metrics ─────────────────────────────────────────
  Future<Map<String, dynamic>> getMetrics() =>
      _get(AppConstants.metricsEndpoint);

  // ─── Level ───────────────────────────────────────────
  Future<Map<String, dynamic>> getLevel() => _get(AppConstants.levelEndpoint);

  // ─── Logs ────────────────────────────────────────────
  Future<Map<String, dynamic>> getLogs({int limit = 100}) =>
      _get('${AppConstants.logsEndpoint}?limit=$limit');

  // ─── Checkpoints ─────────────────────────────────────
  Future<Map<String, dynamic>> getCheckpoints() =>
      _get(AppConstants.checkpointsEndpoint);

  // ─── Infrastructure ──────────────────────────────────
  Future<Map<String, dynamic>> getInfrastructure() =>
      _get(AppConstants.infrastructureEndpoint);

  // ─── Training Control ────────────────────────────────
  Future<Map<String, dynamic>> startTraining({
    int stepsPerChunk = 50,
    int batchSize = 1,
    int gradAccum = 8,
    double learningRate = 5e-5,
    int maxSeqLength = 1024,
  }) => _post(
    AppConstants.trainingStartEndpoint,
    body: {
      'steps_per_chunk': stepsPerChunk,
      'batch_size': batchSize,
      'grad_accum': gradAccum,
      'learning_rate': learningRate,
      'max_seq_length': maxSeqLength,
    },
  );

  Future<Map<String, dynamic>> stopTraining() =>
      _post(AppConstants.trainingStopEndpoint);

  // ─── Chat ────────────────────────────────────────────
  Future<Map<String, dynamic>> sendChat(
    String message, {
    double temperature = 0.7,
    int maxTokens = 512,
  }) => _post(
    AppConstants.chatEndpoint,
    body: {
      'message': message,
      'temperature': temperature,
      'max_tokens': maxTokens,
    },
  );

  // ─── Polling Stream ──────────────────────────────────
  Stream<Map<String, dynamic>> pollEndpoint(
    String endpoint,
    Duration interval,
  ) {
    return Stream.periodic(interval).asyncMap((_) => _get(endpoint));
  }

  void dispose() => _client.close();
}
