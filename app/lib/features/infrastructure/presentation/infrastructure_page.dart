import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/services/api_service.dart';
import '../../../core/widgets/glass_card.dart';

class InfrastructurePage extends StatefulWidget {
  const InfrastructurePage({super.key});

  @override
  State<InfrastructurePage> createState() => _InfrastructurePageState();
}

class _InfrastructurePageState extends State<InfrastructurePage> {
  Map<String, dynamic> _infra = {};
  Map<String, dynamic> _gpu = {};
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _fetchData();
  }

  Future<void> _fetchData() async {
    final api = context.read<ApiService>();
    final results = await Future.wait([api.getInfrastructure(), api.getGpu()]);
    if (!mounted) return;
    setState(() {
      _infra = results[0];
      _gpu = results[1];
      _loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Center(
        child: CircularProgressIndicator(color: AppColors.accent),
      );
    }

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ─── Header ────────────────────────────────
          const Text(
            'Infrastructure',
            style: TextStyle(
              fontSize: 26,
              fontWeight: FontWeight.w800,
              color: AppColors.textPrimary,
              letterSpacing: -0.5,
            ),
          ),
          const SizedBox(height: 6),
          const Text(
            'Sistem bileşenleri ve yapılandırma bilgileri.',
            style: TextStyle(fontSize: 13, color: AppColors.textSecondary),
          ),
          const SizedBox(height: 24),

          // ─── GPU Card ──────────────────────────────
          _buildSection('GPU', Icons.memory_rounded, AppColors.cyan, [
            _InfoRow('Name', _gpu['gpu_name'] ?? 'N/A'),
            _InfoRow('Driver', _gpu['driver_version'] ?? 'N/A'),
            _InfoRow('VRAM Total', '${_gpu['memory_total'] ?? '?'} MB'),
            _InfoRow('VRAM Used', '${_gpu['memory_used'] ?? '?'} MB'),
            _InfoRow('Temperature', '${_gpu['temperature'] ?? '?'}°C'),
            _InfoRow(
              'Power',
              '${_gpu['power_draw'] ?? '?'}W / ${_gpu['power_limit'] ?? '?'}W',
            ),
            _InfoRow('Utilization', '${_gpu['gpu_util'] ?? '?'}%'),
          ]),
          const SizedBox(height: 16),

          // ─── Model Card ────────────────────────────
          _buildSection('Model', Icons.psychology_rounded, AppColors.purple, [
            _InfoRow('Base Model', _str(_infra, 'model.name')),
            _InfoRow('Quantization', _str(_infra, 'model.quantization')),
            _InfoRow('Framework', _str(_infra, 'model.framework')),
            _InfoRow('PEFT', _str(_infra, 'model.peft_version')),
            _InfoRow('TRL', _str(_infra, 'model.trl_version')),
          ]),
          const SizedBox(height: 16),

          // ─── System Card ───────────────────────────
          _buildSection('System', Icons.computer_rounded, AppColors.orange, [
            _InfoRow('OS', _str(_infra, 'system.os')),
            _InfoRow('Python', _str(_infra, 'system.python_version')),
            _InfoRow('PyTorch', _str(_infra, 'system.pytorch_version')),
            _InfoRow('CUDA', _str(_infra, 'system.cuda_version')),
          ]),
          const SizedBox(height: 16),

          // ─── Training Config ───────────────────────
          _buildSection(
            'Training Config',
            Icons.tune_rounded,
            AppColors.green,
            [
              _InfoRow('Strategy', _str(_infra, 'training.strategy')),
              _InfoRow('Steps/Chunk', _str(_infra, 'training.steps_per_chunk')),
              _InfoRow('Batch Size', _str(_infra, 'training.batch_size')),
              _InfoRow('Grad Accum', _str(_infra, 'training.grad_accum')),
              _InfoRow('Learning Rate', _str(_infra, 'training.learning_rate')),
              _InfoRow(
                'Max Seq Length',
                _str(_infra, 'training.max_seq_length'),
              ),
              _InfoRow('Data Samples', _str(_infra, 'training.total_samples')),
            ],
          ),
        ],
      ),
    );
  }

  String _str(Map<String, dynamic> data, String dotPath) {
    final parts = dotPath.split('.');
    dynamic current = data;
    for (final p in parts) {
      if (current is Map) {
        current = current[p];
      } else {
        return 'N/A';
      }
    }
    return current?.toString() ?? 'N/A';
  }

  Widget _buildSection(
    String title,
    IconData icon,
    Color color,
    List<_InfoRow> rows,
  ) {
    return GlassCard(
      glowColor: color,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: color.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Icon(icon, color: color, size: 20),
              ),
              const SizedBox(width: 12),
              Text(
                title,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: AppColors.textPrimary,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          ...rows.map(
            (row) => Padding(
              padding: const EdgeInsets.only(bottom: 10),
              child: Row(
                children: [
                  SizedBox(
                    width: 130,
                    child: Text(
                      row.label,
                      style: const TextStyle(
                        fontSize: 13,
                        color: AppColors.textSecondary,
                      ),
                    ),
                  ),
                  Expanded(
                    child: Text(
                      row.value,
                      style: const TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w500,
                        color: AppColors.textPrimary,
                      ),
                    ),
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

class _InfoRow {
  final String label;
  final String value;
  const _InfoRow(this.label, this.value);
}
