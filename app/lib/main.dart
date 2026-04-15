import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:window_manager/window_manager.dart';
import 'core/theme/app_theme.dart';
import 'core/services/api_service.dart';
import 'core/widgets/app_sidebar.dart';
import 'core/widgets/custom_title_bar.dart';
import 'core/widgets/bubble_background.dart';
import 'features/dashboard/presentation/dashboard_page.dart';
import 'features/training/presentation/training_page.dart';
import 'features/chat/presentation/chat_page.dart';
import 'features/logs/presentation/logs_page.dart';
import 'features/infrastructure/presentation/infrastructure_page.dart';
import 'features/settings/presentation/settings_page.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await windowManager.ensureInitialized();

  const windowOptions = WindowOptions(
    size: Size(1280, 820),
    minimumSize: Size(900, 600),
    center: true,
    backgroundColor: Color(0xFF0A0B10),
    titleBarStyle: TitleBarStyle.hidden,
    title: 'AI Command Center',
  );

  await windowManager.waitUntilReadyToShow(windowOptions, () async {
    await windowManager.show();
    await windowManager.focus();
  });

  runApp(const AiCommandCenterApp());
}

class AiCommandCenterApp extends StatelessWidget {
  const AiCommandCenterApp({super.key});

  @override
  Widget build(BuildContext context) {
    return Provider<ApiService>(
      create: (_) => ApiService(),
      dispose: (_, api) => api.dispose(),
      child: MaterialApp(
        title: 'AI Command Center',
        debugShowCheckedModeBanner: false,
        theme: AppTheme.darkTheme,
        home: const AppShell(),
      ),
    );
  }
}

class AppShell extends StatefulWidget {
  const AppShell({super.key});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int _currentIndex = 0;
  bool _sidebarExpanded = true;

  static const List<Widget> _pages = [
    DashboardPage(),
    TrainingPage(),
    ChatPage(),
    LogsPage(),
    InfrastructurePage(),
    SettingsPage(),
  ];

  @override
  Widget build(BuildContext context) {
    final isNarrow = MediaQuery.of(context).size.width < 800;
    if (isNarrow && _sidebarExpanded) {
      _sidebarExpanded = false;
    }

    return Scaffold(
      body: BubbleBackground(
        child: Column(
          children: [
            const CustomTitleBar(),
            Expanded(
              child: Row(
                children: [
                  AppSidebar(
                    selectedIndex: _currentIndex,
                    onItemTap: (i) => setState(() => _currentIndex = i),
                    isExpanded: _sidebarExpanded,
                    onToggleExpand: () =>
                        setState(() => _sidebarExpanded = !_sidebarExpanded),
                  ),
                  Expanded(
                    child: AnimatedSwitcher(
                      duration: const Duration(milliseconds: 200),
                      child: _pages[_currentIndex],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
