class AppConstants {
  // ─── API ──────────────────────────────────────────────
  static const String apiBaseUrl = 'http://127.0.0.1:3000';

  // Endpoints
  static const String statusEndpoint = '/api/status';
  static const String gpuEndpoint = '/api/gpu';
  static const String metricsEndpoint = '/api/metrics';
  static const String levelEndpoint = '/api/level';
  static const String logsEndpoint = '/api/logs';
  static const String checkpointsEndpoint = '/api/checkpoints';
  static const String infrastructureEndpoint = '/api/infrastructure';
  static const String trainingStartEndpoint = '/api/training/start';
  static const String trainingStopEndpoint = '/api/training/stop';
  static const String chatEndpoint = '/api/chat';

  // ─── Polling Intervals ────────────────────────────────
  static const Duration statusPollInterval = Duration(seconds: 3);
  static const Duration metricsPollInterval = Duration(seconds: 15);
  static const Duration logsPollInterval = Duration(seconds: 5);

  // ─── Level Tiers ──────────────────────────────────────
  static const Map<String, List<int>> levelTiers = {
    'Novice': [1, 14],
    'Apprentice': [15, 29],
    'Specialist': [30, 44],
    'Expert': [45, 59],
    'Master': [60, 74],
    'Grandmaster': [75, 89],
    'Legendary': [90, 99],
  };

  // ─── App Metadata ─────────────────────────────────────
  static const String appName = 'AI Command Center';
  static const String appVersion = '1.0.0';
  static const String appDescription =
      'Topluma Yararlı AI • Training & Monitoring';
}
