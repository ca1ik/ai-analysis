import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/services/api_service.dart';
import '../../../core/widgets/glass_card.dart';

class DashboardPage extends StatefulWidget {
  const DashboardPage({super.key});

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  Map<String, dynamic> _status = {};
  Map<String, dynamic> _gpu = {};
  Map<String, dynamic> _level = {};
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _fetchAll();
    _timer = Timer.periodic(const Duration(seconds: 4), (_) => _fetchAll());
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _fetchAll() async {
    final api = context.read<ApiService>();
    final results = await Future.wait([
      api.getStatus(),
      api.getGpu(),
      api.getLevel(),
    ]);
    if (!mounted) return;
    setState(() {
      _status = results[0];
      _gpu = results[1];
      _level = results[2];
    });
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final crossAxisCount = screenWidth > 1200
        ? 4
        : screenWidth > 800
        ? 3
        : 2;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ─── Header ────────────────────────────────
          _buildHeader(),
          const SizedBox(height: 24),

          // ─── Level Hero + Quick Stats ──────────────
          _buildLevelHero(),
          const SizedBox(height: 24),

          // ─── Stats Grid ────────────────────────────
          GridView.count(
            crossAxisCount: crossAxisCount,
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            mainAxisSpacing: 16,
            crossAxisSpacing: 16,
            childAspectRatio: 1.6,
            children: [
              StatCard(
                icon: Icons.speed_rounded,
                value: '${_gpu['gpu_util'] ?? '--'}%',
                label: 'GPU Utilization',
                color: AppColors.cyan,
              ),
              StatCard(
                icon: Icons.memory_rounded,
                value: '${_gpu['memory_used'] ?? '--'} MB',
                label: 'VRAM Used',
                color: AppColors.purple,
              ),
              StatCard(
                icon: Icons.thermostat_rounded,
                value: '${_gpu['temperature'] ?? '--'}°C',
                label: 'GPU Temperature',
                color: _gpuTempColor,
              ),
              StatCard(
                icon: Icons.bolt_rounded,
                value: '${_gpu['power_draw'] ?? '--'}W',
                label: 'Power Draw',
                color: AppColors.orange,
              ),
            ],
          ),
          const SizedBox(height: 24),

          // ─── Training Progress ─────────────────────
          _buildTrainingProgress(),
          const SizedBox(height: 24),

          // ─── Feature Cards ─────────────────────────
          _buildFeatureCards(),
        ],
      ),
    );
  }

  Color get _gpuTempColor {
    final temp = (_gpu['temperature'] ?? 0) as num;
    if (temp >= 80) return AppColors.red;
    if (temp >= 65) return AppColors.yellow;
    return AppColors.green;
  }

  Widget _buildHeader() {
    return Row(
      children: [
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Command Center',
              style: TextStyle(
                fontSize: 26,
                fontWeight: FontWeight.w800,
                color: AppColors.textPrimary,
                letterSpacing: -0.5,
              ),
            ),
            const SizedBox(height: 4),
            Row(
              children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(
                    color: (_status['training_active'] == true)
                        ? AppColors.green
                        : AppColors.textMuted,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  (_status['training_active'] == true)
                      ? 'Training Active'
                      : 'Idle',
                  style: const TextStyle(
                    fontSize: 13,
                    color: AppColors.textSecondary,
                  ),
                ),
              ],
            ),
          ],
        ),
        const Spacer(),
        StatusBadge(
          label: 'v${_status['model_name'] ?? 'Qwen2.5-7B'}',
          color: AppColors.purple,
        ),
      ],
    );
  }

  Widget _buildLevelHero() {
    final level = (_level['level'] ?? 1) as int;
    final tier = (_level['tier'] ?? 'Novice') as String;
    final xp = (_level['xp'] ?? 0) as num;
    final xpNext = (_level['xp_next'] ?? 100) as num;
    final progress = xpNext > 0 ? (xp / xpNext).clamp(0.0, 1.0) : 0.0;

    return GlassCard(
      glowColor: AppColors.accent,
      child: Row(
        children: [
          LevelRing(level: level, tier: tier, progress: progress.toDouble()),
          const SizedBox(width: 24),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'AI Engineer — $tier',
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 8),
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: progress.toDouble(),
                    backgroundColor: AppColors.bgSurface,
                    valueColor: const AlwaysStoppedAnimation(AppColors.accent),
                    minHeight: 6,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  '${xp.toInt()} / ${xpNext.toInt()} XP',
                  style: const TextStyle(
                    fontSize: 12,
                    color: AppColors.textSecondary,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTrainingProgress() {
    final step = _status['current_step'] ?? 0;
    final totalSteps = _status['total_steps'] ?? 1701;
    final loss = _status['current_loss'];
    final progress = totalSteps > 0 ? (step as num) / (totalSteps as num) : 0.0;

    return GlassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(
                Icons.model_training_rounded,
                color: AppColors.accent,
                size: 20,
              ),
              const SizedBox(width: 8),
              const Text(
                'Training Progress',
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                  color: AppColors.textPrimary,
                ),
              ),
              const Spacer(),
              Text(
                '${(progress * 100).toStringAsFixed(1)}%',
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: AppColors.accent,
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: progress.toDouble().clamp(0.0, 1.0),
              backgroundColor: AppColors.bgSurface,
              valueColor: const AlwaysStoppedAnimation(AppColors.accent),
              minHeight: 8,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              _miniStat('Step', '$step / $totalSteps'),
              const SizedBox(width: 24),
              _miniStat(
                'Loss',
                loss != null ? (loss as num).toStringAsFixed(4) : '--',
              ),
              const SizedBox(width: 24),
              _miniStat('GPU', '${_gpu['gpu_name'] ?? 'RTX 5070'}'),
            ],
          ),
        ],
      ),
    );
  }

  Widget _miniStat(String label, String value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(fontSize: 11, color: AppColors.textMuted),
        ),
        const SizedBox(height: 2),
        Text(
          value,
          style: const TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w600,
            color: AppColors.textPrimary,
          ),
        ),
      ],
    );
  }

  Widget _buildFeatureCards() {
    final features = [
      _FeatureItem(
        Icons.smart_toy_rounded,
        'AI Chat',
        'Trained model ile sohbet',
        AppColors.accent,
      ),
      _FeatureItem(
        Icons.model_training_rounded,
        'Training Control',
        'Micro-chunk training yönetimi',
        AppColors.purple,
      ),
      _FeatureItem(
        Icons.terminal_rounded,
        'Live Logs',
        'Gerçek zamanlı log takibi',
        AppColors.blue,
      ),
      _FeatureItem(
        Icons.memory_rounded,
        'Infrastructure',
        'GPU, VRAM, CUDA durumu',
        AppColors.orange,
      ),
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Quick Access',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w700,
            color: AppColors.textPrimary,
          ),
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 12,
          runSpacing: 12,
          children: features.map((f) => _buildFeatureCard(f)).toList(),
        ),
      ],
    );
  }

  Widget _buildFeatureCard(_FeatureItem item) {
    return SizedBox(
      width: 200,
      child: GlassCard(
        padding: const EdgeInsets.all(16),
        glowColor: item.color,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: item.color.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(item.icon, color: item.color, size: 22),
            ),
            const SizedBox(height: 12),
            Text(
              item.title,
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: AppColors.textPrimary,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              item.description,
              style: const TextStyle(
                fontSize: 11,
                color: AppColors.textSecondary,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _FeatureItem {
  final IconData icon;
  final String title;
  final String description;
  final Color color;
  const _FeatureItem(this.icon, this.title, this.description, this.color);
}
