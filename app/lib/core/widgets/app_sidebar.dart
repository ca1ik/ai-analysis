import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import '../../core/theme/app_theme.dart';

/// Navigation item data
class NavItem {
  final IconData icon;
  final String label;
  final String? badge;

  const NavItem({required this.icon, required this.label, this.badge});
}

/// Multi AI Studio style sidebar
class AppSidebar extends StatelessWidget {
  final int selectedIndex;
  final ValueChanged<int> onItemTap;
  final bool isExpanded;
  final VoidCallback onToggleExpand;

  static const List<NavItem> items = [
    NavItem(icon: Icons.dashboard_rounded, label: 'Dashboard'),
    NavItem(icon: Icons.model_training_rounded, label: 'Training'),
    NavItem(icon: Icons.smart_toy_rounded, label: 'AI Chat'),
    NavItem(icon: Icons.terminal_rounded, label: 'Logs'),
    NavItem(icon: Icons.memory_rounded, label: 'Infrastructure'),
    NavItem(icon: Icons.settings_rounded, label: 'Settings'),
  ];

  static const List<String> sectionHeaders = [
    '', // Dashboard — no header
    'TRAINING',
    '',
    'SYSTEM',
    '',
    '',
  ];

  const AppSidebar({
    super.key,
    required this.selectedIndex,
    required this.onItemTap,
    required this.isExpanded,
    required this.onToggleExpand,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      curve: Curves.easeInOut,
      width: isExpanded ? 230 : 68,
      decoration: const BoxDecoration(
        color: AppColors.bgSidebar,
        border: Border(right: BorderSide(color: AppColors.border)),
      ),
      child: Column(
        children: [
          // ─── Logo Header ──────────────────────────
          Container(
            height: 64,
            padding: const EdgeInsets.symmetric(horizontal: 14),
            child: Row(
              children: [
                Container(
                  width: 36,
                  height: 36,
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      colors: [AppColors.accent, AppColors.cyan],
                    ),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: const Icon(
                    Icons.hub_rounded,
                    color: AppColors.textOnAccent,
                    size: 20,
                  ),
                ),
                if (isExpanded) ...[
                  const SizedBox(width: 12),
                  const Expanded(
                    child: Text(
                      'AI Center',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                        color: AppColors.textPrimary,
                        letterSpacing: -0.3,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ],
              ],
            ),
          ),
          const Divider(height: 1),

          // ─── Nav Items ────────────────────────────
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 8),
              itemCount: items.length,
              itemBuilder: (context, index) {
                final showHeader =
                    sectionHeaders[index].isNotEmpty &&
                    (index == 0 ||
                        sectionHeaders[index] != sectionHeaders[index - 1]);

                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (showHeader && isExpanded)
                      Padding(
                        padding: const EdgeInsets.only(
                          left: 8,
                          top: 16,
                          bottom: 6,
                        ),
                        child: Text(
                          sectionHeaders[index],
                          style: const TextStyle(
                            fontSize: 10,
                            fontWeight: FontWeight.w700,
                            color: AppColors.textMuted,
                            letterSpacing: 1.5,
                          ),
                        ),
                      ),
                    _NavTile(
                      item: items[index],
                      isSelected: selectedIndex == index,
                      isExpanded: isExpanded,
                      onTap: () => onItemTap(index),
                    ),
                  ],
                );
              },
            ),
          ),

          // ─── Web App Button ───────────────────────
          const Divider(height: 1),
          _WebAppButton(isExpanded: isExpanded),

          // ─── Collapse Toggle ──────────────────────
          const Divider(height: 1),
          InkWell(
            onTap: onToggleExpand,
            child: Container(
              height: 48,
              padding: const EdgeInsets.symmetric(horizontal: 14),
              child: Row(
                mainAxisAlignment: isExpanded
                    ? MainAxisAlignment.end
                    : MainAxisAlignment.center,
                children: [
                  Icon(
                    isExpanded
                        ? Icons.chevron_left_rounded
                        : Icons.chevron_right_rounded,
                    color: AppColors.textMuted,
                    size: 20,
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

class _NavTile extends StatefulWidget {
  final NavItem item;
  final bool isSelected;
  final bool isExpanded;
  final VoidCallback onTap;

  const _NavTile({
    required this.item,
    required this.isSelected,
    required this.isExpanded,
    required this.onTap,
  });

  @override
  State<_NavTile> createState() => _NavTileState();
}

class _NavTileState extends State<_NavTile> {
  bool _hovering = false;

  @override
  Widget build(BuildContext context) {
    final color = widget.isSelected
        ? AppColors.accent
        : AppColors.textSecondary;
    final bg = widget.isSelected
        ? AppColors.accentGlow
        : _hovering
        ? AppColors.bgCardHover
        : Colors.transparent;

    return MouseRegion(
      onEnter: (_) => setState(() => _hovering = true),
      onExit: (_) => setState(() => _hovering = false),
      cursor: SystemMouseCursors.click,
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 150),
          margin: const EdgeInsets.symmetric(vertical: 2),
          padding: EdgeInsets.symmetric(
            horizontal: widget.isExpanded ? 12 : 0,
            vertical: 10,
          ),
          decoration: BoxDecoration(
            color: bg,
            borderRadius: BorderRadius.circular(10),
            border: widget.isSelected
                ? Border.all(color: AppColors.accent.withValues(alpha: 0.2))
                : null,
          ),
          child: Row(
            mainAxisAlignment: widget.isExpanded
                ? MainAxisAlignment.start
                : MainAxisAlignment.center,
            children: [
              Icon(widget.item.icon, color: color, size: 20),
              if (widget.isExpanded) ...[
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    widget.item.label,
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: widget.isSelected
                          ? FontWeight.w600
                          : FontWeight.w500,
                      color: color,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                if (widget.item.badge != null)
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 6,
                      vertical: 2,
                    ),
                    decoration: BoxDecoration(
                      color: AppColors.accent.withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      widget.item.badge!,
                      style: const TextStyle(
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                        color: AppColors.accent,
                      ),
                    ),
                  ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _WebAppButton extends StatefulWidget {
  final bool isExpanded;
  const _WebAppButton({required this.isExpanded});

  @override
  State<_WebAppButton> createState() => _WebAppButtonState();
}

class _WebAppButtonState extends State<_WebAppButton> {
  bool _hovering = false;

  Future<void> _openWebApp() async {
    final uri = Uri.parse('http://127.0.0.1:3000');
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) => setState(() => _hovering = true),
      onExit: (_) => setState(() => _hovering = false),
      cursor: SystemMouseCursors.click,
      child: GestureDetector(
        onTap: _openWebApp,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 150),
          margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
          padding: EdgeInsets.symmetric(
            horizontal: widget.isExpanded ? 12 : 0,
            vertical: 10,
          ),
          decoration: BoxDecoration(
            gradient: _hovering
                ? const LinearGradient(
                    colors: [Color(0x20FF6B6B), Color(0x20FF9F43)],
                  )
                : null,
            color: _hovering ? null : Colors.transparent,
            borderRadius: BorderRadius.circular(10),
            border: _hovering
                ? Border.all(color: AppColors.orange.withValues(alpha: 0.3))
                : null,
          ),
          child: Row(
            mainAxisAlignment: widget.isExpanded
                ? MainAxisAlignment.start
                : MainAxisAlignment.center,
            children: [
              Icon(
                Icons.language_rounded,
                color: _hovering ? AppColors.orange : AppColors.textSecondary,
                size: 20,
              ),
              if (widget.isExpanded) ...[
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    'Web App',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w500,
                      color: _hovering
                          ? AppColors.orange
                          : AppColors.textSecondary,
                    ),
                  ),
                ),
                Icon(
                  Icons.open_in_new_rounded,
                  color: _hovering
                      ? AppColors.orange.withValues(alpha: 0.7)
                      : AppColors.textMuted,
                  size: 14,
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
