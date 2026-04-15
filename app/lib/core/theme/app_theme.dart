import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppColors {
  // ─── Base ──────────────────────────────────────────────
  static const Color bgPrimary = Color(0xFF0A0B10);
  static const Color bgSecondary = Color(0xFF0F1018);
  static const Color bgCard = Color(0xFF151621);
  static const Color bgCardHover = Color(0xFF1A1B2E);
  static const Color bgSidebar = Color(0xFF0C0D14);
  static const Color bgInput = Color(0xFF12131F);
  static const Color bgSurface = Color(0xFF1E1F30);

  // ─── Accent (Cyan/Teal from Multi AI Studio) ──────────
  static const Color accent = Color(0xFF00E5C3);
  static const Color accentDark = Color(0xFF00B89C);
  static const Color accentGlow = Color(0x3300E5C3);
  static const Color accentLight = Color(0xFF5EFCE8);

  // ─── Semantic ─────────────────────────────────────────
  static const Color green = Color(0xFF00D68F);
  static const Color red = Color(0xFFFF6B6B);
  static const Color yellow = Color(0xFFFECA57);
  static const Color orange = Color(0xFFF0932B);
  static const Color purple = Color(0xFF6C5CE7);
  static const Color cyan = Color(0xFF00CEC9);
  static const Color pink = Color(0xFFE056FD);
  static const Color blue = Color(0xFF45B7D1);

  // ─── Text ─────────────────────────────────────────────
  static const Color textPrimary = Color(0xFFE8E8F0);
  static const Color textSecondary = Color(0xFF8888A8);
  static const Color textMuted = Color(0xFF555570);
  static const Color textOnAccent = Color(0xFF0A0B10);

  // ─── Border ───────────────────────────────────────────
  static const Color border = Color(0x15FFFFFF);
  static const Color borderLight = Color(0x25FFFFFF);
}

class AppTheme {
  static ThemeData get darkTheme {
    return ThemeData(
      brightness: Brightness.dark,
      scaffoldBackgroundColor: AppColors.bgPrimary,
      colorScheme: const ColorScheme.dark(
        primary: AppColors.accent,
        secondary: AppColors.accentDark,
        surface: AppColors.bgCard,
        error: AppColors.red,
      ),
      textTheme: GoogleFonts.interTextTheme(ThemeData.dark().textTheme).apply(
        bodyColor: AppColors.textPrimary,
        displayColor: AppColors.textPrimary,
      ),
      cardTheme: CardThemeData(
        color: AppColors.bgCard,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(14),
          side: const BorderSide(color: AppColors.border),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: AppColors.bgInput,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: AppColors.accent, width: 1.5),
        ),
        contentPadding: const EdgeInsets.symmetric(
          horizontal: 16,
          vertical: 14,
        ),
        hintStyle: const TextStyle(color: AppColors.textMuted, fontSize: 14),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: AppColors.accent,
          foregroundColor: AppColors.textOnAccent,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
          ),
          textStyle: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14),
        ),
      ),
      dividerTheme: const DividerThemeData(
        color: AppColors.border,
        thickness: 1,
      ),
    );
  }
}
