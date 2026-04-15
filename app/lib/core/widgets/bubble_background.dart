import 'dart:math';
import 'package:flutter/material.dart';
import '../../core/theme/app_theme.dart';

class BubbleBackground extends StatefulWidget {
  final Widget child;
  const BubbleBackground({super.key, required this.child});

  @override
  State<BubbleBackground> createState() => _BubbleBackgroundState();
}

class _BubbleBackgroundState extends State<BubbleBackground>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late List<_Bubble> _bubbles;
  final _random = Random();

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 1),
    )..repeat();
    _bubbles = List.generate(12, (_) => _createBubble());
  }

  _Bubble _createBubble() {
    return _Bubble(
      x: _random.nextDouble(),
      y: _random.nextDouble(),
      maxRadius: 4 + _random.nextDouble() * 18,
      phase: _random.nextDouble() * 2 * pi,
      speed: 0.15 + _random.nextDouble() * 0.35,
      color: [
        AppColors.accent,
        AppColors.cyan,
        AppColors.purple,
        AppColors.blue,
      ][_random.nextInt(4)],
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // Bubble layer
        Positioned.fill(
          child: AnimatedBuilder(
            animation: _controller,
            builder: (context, _) => CustomPaint(
              painter: _BubblePainter(
                bubbles: _bubbles,
                time: DateTime.now().millisecondsSinceEpoch / 1000.0,
              ),
            ),
          ),
        ),
        // Content on top
        widget.child,
      ],
    );
  }
}

class _Bubble {
  final double x;
  final double y;
  final double maxRadius;
  final double phase;
  final double speed;
  final Color color;

  const _Bubble({
    required this.x,
    required this.y,
    required this.maxRadius,
    required this.phase,
    required this.speed,
    required this.color,
  });
}

class _BubblePainter extends CustomPainter {
  final List<_Bubble> bubbles;
  final double time;

  _BubblePainter({required this.bubbles, required this.time});

  @override
  void paint(Canvas canvas, Size size) {
    for (final b in bubbles) {
      // Slow sine wave for grow/shrink cycle
      final cycle = sin(time * b.speed + b.phase);
      // Map -1..1 to 0..1 (0 = invisible, 1 = max radius)
      final t = (cycle + 1) / 2;
      final radius = b.maxRadius * t;
      if (radius < 0.5) continue;

      // Very subtle opacity: max 0.06
      final opacity = 0.06 * t;

      final paint = Paint()
        ..color = b.color.withValues(alpha: opacity)
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, radius * 0.8);

      canvas.drawCircle(
        Offset(b.x * size.width, b.y * size.height),
        radius,
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _BubblePainter old) => true;
}
