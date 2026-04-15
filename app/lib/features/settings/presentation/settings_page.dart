import 'package:flutter/material.dart';
import '../../../core/theme/app_theme.dart';
import '../../../core/widgets/glass_card.dart';
import '../../../core/constants/app_constants.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key});

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  String _apiUrl = AppConstants.apiBaseUrl;
  bool _darkMode = true;
  bool _autoRefresh = true;
  int _refreshInterval = 4;

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ─── Header ────────────────────────────────
          const Text(
            'Settings',
            style: TextStyle(
              fontSize: 26,
              fontWeight: FontWeight.w800,
              color: AppColors.textPrimary,
              letterSpacing: -0.5,
            ),
          ),
          const SizedBox(height: 6),
          const Text(
            'Uygulama yapılandırması ve tercihler.',
            style: TextStyle(fontSize: 13, color: AppColors.textSecondary),
          ),
          const SizedBox(height: 24),

          // ─── Connection ────────────────────────────
          GlassCard(
            glowColor: AppColors.accent,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _sectionTitle(
                  'Connection',
                  Icons.link_rounded,
                  AppColors.accent,
                ),
                const SizedBox(height: 16),
                _buildTextField(
                  label: 'API Base URL',
                  value: _apiUrl,
                  onChanged: (v) => setState(() => _apiUrl = v),
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    const Text(
                      'Auto Refresh',
                      style: TextStyle(
                        fontSize: 13,
                        color: AppColors.textSecondary,
                      ),
                    ),
                    const Spacer(),
                    Switch(
                      value: _autoRefresh,
                      onChanged: (v) => setState(() => _autoRefresh = v),
                      activeTrackColor: AppColors.accent,
                    ),
                  ],
                ),
                if (_autoRefresh) ...[
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      const Text(
                        'Refresh Interval',
                        style: TextStyle(
                          fontSize: 13,
                          color: AppColors.textSecondary,
                        ),
                      ),
                      const Spacer(),
                      SizedBox(
                        width: 200,
                        child: Slider(
                          value: _refreshInterval.toDouble(),
                          min: 1,
                          max: 30,
                          divisions: 29,
                          activeColor: AppColors.accent,
                          label: '${_refreshInterval}s',
                          onChanged: (v) =>
                              setState(() => _refreshInterval = v.toInt()),
                        ),
                      ),
                      Text(
                        '${_refreshInterval}s',
                        style: const TextStyle(
                          fontSize: 13,
                          color: AppColors.textPrimary,
                        ),
                      ),
                    ],
                  ),
                ],
              ],
            ),
          ),
          const SizedBox(height: 16),

          // ─── Appearance ────────────────────────────
          GlassCard(
            glowColor: AppColors.purple,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _sectionTitle(
                  'Appearance',
                  Icons.palette_rounded,
                  AppColors.purple,
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    const Text(
                      'Dark Mode',
                      style: TextStyle(
                        fontSize: 13,
                        color: AppColors.textSecondary,
                      ),
                    ),
                    const Spacer(),
                    Switch(
                      value: _darkMode,
                      onChanged: (v) => setState(() => _darkMode = v),
                      activeTrackColor: AppColors.accent,
                    ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // ─── About ─────────────────────────────────
          GlassCard(
            glowColor: AppColors.blue,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _sectionTitle('About', Icons.info_rounded, AppColors.blue),
                const SizedBox(height: 16),
                _aboutRow('App', AppConstants.appName),
                _aboutRow('Version', AppConstants.appVersion),
                _aboutRow('Description', AppConstants.appDescription),
                _aboutRow('Platform', Theme.of(context).platform.name),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _sectionTitle(String title, IconData icon, Color color) {
    return Row(
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
    );
  }

  Widget _buildTextField({
    required String label,
    required String value,
    required ValueChanged<String> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(fontSize: 12, color: AppColors.textSecondary),
        ),
        const SizedBox(height: 6),
        TextFormField(
          initialValue: value,
          onChanged: onChanged,
          style: const TextStyle(fontSize: 13, color: AppColors.textPrimary),
        ),
      ],
    );
  }

  Widget _aboutRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        children: [
          SizedBox(
            width: 100,
            child: Text(
              label,
              style: const TextStyle(
                fontSize: 13,
                color: AppColors.textSecondary,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w500,
                color: AppColors.textPrimary,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
