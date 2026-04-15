import 'dart:io';
import 'package:flutter/material.dart';
import 'package:window_manager/window_manager.dart';
import '../../core/theme/app_theme.dart';

class CustomTitleBar extends StatefulWidget {
  const CustomTitleBar({super.key});

  @override
  State<CustomTitleBar> createState() => _CustomTitleBarState();
}

class _CustomTitleBarState extends State<CustomTitleBar> with WindowListener {
  bool _isMaximized = false;

  @override
  void initState() {
    super.initState();
    windowManager.addListener(this);
    _checkMaximized();
  }

  @override
  void dispose() {
    windowManager.removeListener(this);
    super.dispose();
  }

  Future<void> _checkMaximized() async {
    final m = await windowManager.isMaximized();
    if (mounted) setState(() => _isMaximized = m);
  }

  @override
  void onWindowMaximize() => setState(() => _isMaximized = true);
  @override
  void onWindowUnmaximize() => setState(() => _isMaximized = false);

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onPanStart: (_) => windowManager.startDragging(),
      onDoubleTap: () async {
        if (_isMaximized) {
          await windowManager.unmaximize();
        } else {
          await windowManager.maximize();
        }
      },
      child: Container(
        height: 38,
        color: AppColors.bgSidebar,
        child: Row(
          children: [
            const SizedBox(width: 14),
            // App icon + title
            Container(
              width: 22,
              height: 22,
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [AppColors.accent, AppColors.cyan],
                ),
                borderRadius: BorderRadius.circular(6),
              ),
              child: const Icon(
                Icons.hub_rounded,
                color: AppColors.textOnAccent,
                size: 12,
              ),
            ),
            const SizedBox(width: 10),
            const Text(
              'AI Command Center',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: AppColors.textSecondary,
                letterSpacing: 0.3,
              ),
            ),
            // Drag area fills the rest
            const Expanded(child: SizedBox.shrink()),
            // Window controls
            _TitleButton(
              icon: Icons.horizontal_rule_rounded,
              hoverColor: AppColors.textSecondary.withValues(alpha: 0.12),
              iconColor: AppColors.textSecondary,
              onTap: () => windowManager.minimize(),
              tooltip: 'Minimize',
            ),
            _TitleButton(
              icon: _isMaximized
                  ? Icons.filter_none_rounded
                  : Icons.crop_square_rounded,
              hoverColor: AppColors.textSecondary.withValues(alpha: 0.12),
              iconColor: AppColors.textSecondary,
              iconSize: _isMaximized ? 13 : 15,
              onTap: () async {
                if (_isMaximized) {
                  await windowManager.unmaximize();
                } else {
                  await windowManager.maximize();
                }
              },
              tooltip: _isMaximized ? 'Restore' : 'Maximize',
            ),
            _TitleButton(
              icon: Icons.close_rounded,
              hoverColor: const Color(0xFFE81123),
              iconColor: AppColors.textSecondary,
              hoverIconColor: Colors.white,
              onTap: () => exit(0),
              tooltip: 'Close',
            ),
          ],
        ),
      ),
    );
  }
}

class _TitleButton extends StatefulWidget {
  final IconData icon;
  final Color hoverColor;
  final Color iconColor;
  final Color? hoverIconColor;
  final double iconSize;
  final VoidCallback onTap;
  final String tooltip;

  const _TitleButton({
    required this.icon,
    required this.hoverColor,
    required this.iconColor,
    this.hoverIconColor,
    this.iconSize = 15,
    required this.onTap,
    required this.tooltip,
  });

  @override
  State<_TitleButton> createState() => _TitleButtonState();
}

class _TitleButtonState extends State<_TitleButton> {
  bool _hovering = false;

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: widget.tooltip,
      waitDuration: const Duration(milliseconds: 600),
      child: MouseRegion(
        onEnter: (_) => setState(() => _hovering = true),
        onExit: (_) => setState(() => _hovering = false),
        child: GestureDetector(
          onTap: widget.onTap,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 120),
            width: 46,
            height: 38,
            color: _hovering ? widget.hoverColor : Colors.transparent,
            alignment: Alignment.center,
            child: Icon(
              widget.icon,
              size: widget.iconSize,
              color: _hovering
                  ? (widget.hoverIconColor ?? widget.iconColor)
                  : widget.iconColor,
            ),
          ),
        ),
      ),
    );
  }
}
