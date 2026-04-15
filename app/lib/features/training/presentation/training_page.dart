import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/services/api_service.dart';
import '../../../core/widgets/glass_card.dart';

class TrainingPage extends StatefulWidget {
  const TrainingPage({super.key});

  @override
  State<TrainingPage> createState() => _TrainingPageState();
}

class _TrainingPageState extends State<TrainingPage> {
  Map<String, dynamic> _status = {};
  Map<String, dynamic> _checkpoints = {};
  bool _isLoading = false;
  Timer? _timer;

  // Config defaults
  int _stepsPerChunk = 50;
  int _batchSize = 1;
  int _gradAccum = 8;
  double _learningRate = 5e-5;
  int _maxSeqLength = 1024;

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
    final results = await Future.wait([api.getStatus(), api.getCheckpoints()]);
    if (!mounted) return;
    setState(() {
      _status = results[0];
      _checkpoints = results[1];
    });
  }

  Future<void> _startTraining() async {
    setState(() => _isLoading = true);
    final api = context.read<ApiService>();
    await api.startTraining(
      stepsPerChunk: _stepsPerChunk,
      batchSize: _batchSize,
      gradAccum: _gradAccum,
      learningRate: _learningRate,
      maxSeqLength: _maxSeqLength,
    );
    await _fetchAll();
    setState(() => _isLoading = false);
  }

  Future<void> _stopTraining() async {
    setState(() => _isLoading = true);
    final api = context.read<ApiService>();
    await api.stopTraining();
    await _fetchAll();
    setState(() => _isLoading = false);
  }

  @override
  Widget build(BuildContext context) {
    final isTraining = _status['training_active'] == true;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ─── Header ────────────────────────────────
          Row(
            children: [
              const Text(
                'Training Control',
                style: TextStyle(
                  fontSize: 26,
                  fontWeight: FontWeight.w800,
                  color: AppColors.textPrimary,
                  letterSpacing: -0.5,
                ),
              ),
              const Spacer(),
              StatusBadge(
                label: isTraining ? 'ACTIVE' : 'IDLE',
                color: isTraining ? AppColors.green : AppColors.textMuted,
              ),
            ],
          ),
          const SizedBox(height: 24),

          // ─── Controls ──────────────────────────────
          GlassCard(
            glowColor: isTraining ? AppColors.green : null,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Micro-Chunk Training',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 4),
                const Text(
                  'Her chunk ayrı process olarak çalışır — BSOD\'a karşı güvenli.',
                  style: TextStyle(
                    fontSize: 12,
                    color: AppColors.textSecondary,
                  ),
                ),
                const SizedBox(height: 20),

                // Config Row
                Wrap(
                  spacing: 16,
                  runSpacing: 16,
                  children: [
                    _buildConfigField(
                      'Steps/Chunk',
                      '$_stepsPerChunk',
                      (v) => _stepsPerChunk = int.tryParse(v) ?? 50,
                    ),
                    _buildConfigField(
                      'Batch Size',
                      '$_batchSize',
                      (v) => _batchSize = int.tryParse(v) ?? 1,
                    ),
                    _buildConfigField(
                      'Grad Accum',
                      '$_gradAccum',
                      (v) => _gradAccum = int.tryParse(v) ?? 8,
                    ),
                    _buildConfigField(
                      'Learning Rate',
                      '$_learningRate',
                      (v) => _learningRate = double.tryParse(v) ?? 5e-5,
                    ),
                    _buildConfigField(
                      'Max Length',
                      '$_maxSeqLength',
                      (v) => _maxSeqLength = int.tryParse(v) ?? 1024,
                    ),
                  ],
                ),
                const SizedBox(height: 20),

                // Action Buttons
                Row(
                  children: [
                    ElevatedButton.icon(
                      onPressed: _isLoading
                          ? null
                          : isTraining
                          ? _stopTraining
                          : _startTraining,
                      icon: Icon(
                        isTraining
                            ? Icons.stop_rounded
                            : Icons.play_arrow_rounded,
                        size: 20,
                      ),
                      label: Text(
                        isTraining ? 'Stop Training' : 'Start Training',
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: isTraining
                            ? AppColors.red
                            : AppColors.accent,
                        foregroundColor: isTraining
                            ? Colors.white
                            : AppColors.textOnAccent,
                      ),
                    ),
                    if (_isLoading) ...[
                      const SizedBox(width: 12),
                      const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: AppColors.accent,
                        ),
                      ),
                    ],
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // ─── Progress ──────────────────────────────
          _buildProgressSection(),
          const SizedBox(height: 24),

          // ─── Checkpoints ───────────────────────────
          _buildCheckpointsSection(),
        ],
      ),
    );
  }

  Widget _buildConfigField(
    String label,
    String initialValue,
    ValueChanged<String> onChanged,
  ) {
    return SizedBox(
      width: 150,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: const TextStyle(
              fontSize: 11,
              color: AppColors.textSecondary,
            ),
          ),
          const SizedBox(height: 6),
          TextFormField(
            initialValue: initialValue,
            onChanged: onChanged,
            style: const TextStyle(fontSize: 13, color: AppColors.textPrimary),
            decoration: const InputDecoration(
              isDense: true,
              contentPadding: EdgeInsets.symmetric(
                horizontal: 12,
                vertical: 10,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildProgressSection() {
    final step = _status['current_step'] ?? 0;
    final totalSteps = _status['total_steps'] ?? 1701;
    final loss = _status['current_loss'];
    final epoch = _status['current_epoch'] ?? 0;
    final progress = totalSteps > 0 ? (step as num) / (totalSteps as num) : 0.0;

    return GlassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(
                Icons.show_chart_rounded,
                color: AppColors.accent,
                size: 20,
              ),
              const SizedBox(width: 8),
              const Text(
                'Current Progress',
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                  color: AppColors.textPrimary,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: progress.toDouble().clamp(0.0, 1.0),
              backgroundColor: AppColors.bgSurface,
              valueColor: const AlwaysStoppedAnimation(AppColors.accent),
              minHeight: 8,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _infoChip('Step', '$step / $totalSteps'),
              _infoChip('Epoch', '$epoch'),
              _infoChip(
                'Loss',
                loss != null ? (loss as num).toStringAsFixed(4) : '--',
              ),
              _infoChip('Progress', '${(progress * 100).toStringAsFixed(1)}%'),
            ],
          ),
        ],
      ),
    );
  }

  Widget _infoChip(String label, String value) {
    return Column(
      children: [
        Text(
          value,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w700,
            color: AppColors.textPrimary,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          label,
          style: const TextStyle(fontSize: 11, color: AppColors.textMuted),
        ),
      ],
    );
  }

  Widget _buildCheckpointsSection() {
    final checkpoints = _checkpoints['checkpoints'] as List? ?? [];

    return GlassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.save_rounded, color: AppColors.purple, size: 20),
              const SizedBox(width: 8),
              const Text(
                'Checkpoints',
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                  color: AppColors.textPrimary,
                ),
              ),
              const Spacer(),
              Text(
                '${checkpoints.length} saved',
                style: const TextStyle(
                  fontSize: 12,
                  color: AppColors.textSecondary,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          if (checkpoints.isEmpty)
            const Padding(
              padding: EdgeInsets.all(12),
              child: Text(
                'Henüz checkpoint kaydedilmedi.',
                style: TextStyle(color: AppColors.textMuted, fontSize: 13),
              ),
            )
          else
            ...checkpoints.map(
              (cp) => Container(
                margin: const EdgeInsets.only(bottom: 8),
                padding: const EdgeInsets.symmetric(
                  horizontal: 14,
                  vertical: 10,
                ),
                decoration: BoxDecoration(
                  color: AppColors.bgInput,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: AppColors.border),
                ),
                child: Row(
                  children: [
                    const Icon(
                      Icons.folder_rounded,
                      color: AppColors.accent,
                      size: 16,
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        cp['name']?.toString() ?? cp.toString(),
                        style: const TextStyle(
                          fontSize: 13,
                          color: AppColors.textPrimary,
                        ),
                      ),
                    ),
                    if (cp is Map && cp['step'] != null)
                      StatusBadge(
                        label: 'Step ${cp['step']}',
                        color: AppColors.cyan,
                      ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}
